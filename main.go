package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type runtimeConfig struct {
	Host                string `json:"listen_host"`
	Port                string `json:"listen_port"`
	LocalAuthToken      string `json:"local_auth_token"`
	UpstreamBaseURL     string `json:"upstream_base_url"`
	UpstreamToken       string `json:"upstream_token"`
	FastMode            bool   `json:"fast_mode"`
	ServiceTier         string `json:"service_tier"`
	ReasoningEffort     string `json:"reasoning_effort"`
	ContextWindowLength int    `json:"context_window_length"`
	Model               string `json:"upstream_model"`
	CacheID             string
	ConfigPath          string
}

type fileConfig struct {
	ListenHost          string `json:"listen_host"`
	ListenPort          string `json:"listen_port"`
	LocalAuthToken      string `json:"local_auth_token"`
	UpstreamBaseURL     string `json:"upstream_base_url"`
	UpstreamToken       string `json:"upstream_token"`
	FastMode            *bool  `json:"fast_mode,omitempty"`
	Fastmode            *bool  `json:"fastmode,omitempty"`
	ReasoningEffort     string `json:"reasoning_effort"`
	ContextWindowLength int    `json:"context_window_length"`
	UpstreamModel       string `json:"upstream_model"`
}

type sessionState struct {
	StateKey           string    `json:"state_key"`
	SessionID          string    `json:"session_id"`
	Model              string    `json:"model"`
	PromptCacheKey     string    `json:"prompt_cache_key"`
	ResponseID         string    `json:"response_id"`
	MessageHashes      []string  `json:"message_hashes"`
	TotalMessageTokens int       `json:"total_message_tokens"`
	Synthetic          bool      `json:"synthetic"`
	UpdatedAt          time.Time `json:"updated_at"`
}

type sessionStore struct {
	mu   sync.RWMutex
	data map[string]sessionState
}

type persistedGatewayState struct {
	CacheID  string         `json:"cache_id"`
	Sessions []sessionState `json:"sessions"`
}

type statePersistence struct {
	mu         sync.Mutex
	path       string
	cacheID    string
	maxEntries int
	maxAge     time.Duration
}

func (s *sessionStore) Get(key string) (sessionState, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	value, ok := s.data[key]
	return value, ok
}

func (s *sessionStore) Set(key string, value sessionState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	value.UpdatedAt = time.Now()
	s.data[key] = value
}

func (s *sessionStore) Replace(values map[string]sessionState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = values
}

func (s *sessionStore) Snapshot() map[string]sessionState {
	s.mu.RLock()
	defer s.mu.RUnlock()

	snapshot := make(map[string]sessionState, len(s.data))
	for key, value := range s.data {
		snapshot[key] = cloneSessionState(value)
	}
	return snapshot
}

func (s *sessionStore) FindSyntheticPrefix(promptCacheKey string, messageHashes []string) (string, sessionState, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	bestKey := ""
	var bestState sessionState
	bestLen := -1
	for key, state := range s.data {
		if !state.Synthetic || state.PromptCacheKey != promptCacheKey {
			continue
		}
		if len(state.MessageHashes) == 0 || len(state.MessageHashes) >= len(messageHashes) {
			continue
		}
		if !isHashPrefix(state.MessageHashes, messageHashes) {
			continue
		}
		if len(state.MessageHashes) > bestLen {
			bestKey = key
			bestState = state
			bestLen = len(state.MessageHashes)
		}
	}
	if bestKey == "" {
		return "", sessionState{}, false
	}
	return bestKey, bestState, true
}

var sessions = &sessionStore{data: map[string]sessionState{}}
var persistedState = &statePersistence{
	maxEntries: 1024,
	maxAge:     24 * time.Hour,
}
var responsesHTTPClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       300 * time.Second,
		TLSHandshakeTimeout:   300 * time.Second,
		ExpectContinueTimeout: 300 * time.Second,
		ResponseHeaderTimeout: 300 * time.Second,
	},
}

func main() {
	cfg, err := loadRuntimeConfig()
	if err != nil {
		log.Fatal(err)
	}
	if err := ensureGeneratedClaudeSettings(cfg); err != nil {
		log.Fatal(err)
	}

	addr := cfg.Host + ":" + cfg.Port
	server := &http.Server{
		Addr:              addr,
		Handler:           newHandler(cfg),
		ReadHeaderTimeout: 300 * time.Second,
	}

	log.Printf("cc-adapter listening on http://%s", addr)
	log.Fatal(server.ListenAndServe())
}

func newHandler(cfg *runtimeConfig) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if !authenticate(cfg, w, r) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": cfg.Model, "object": "model", "created": 1677610602, "owned_by": "openai"},
				{"id": "claude-opus-4-6", "object": "model", "created": 1677610602, "owned_by": "openai"},
				{"id": "claude-opus-4-6[1m]", "object": "model", "created": 1677610602, "owned_by": "openai"},
			},
		})
	})
	mux.HandleFunc("/v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		if !authenticate(cfg, w, r) {
			return
		}
		body, err := decodeBody(r)
		if err != nil {
			writeAnthropicError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"input_tokens": approximateTokens(body),
		})
	})
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		if !authenticate(cfg, w, r) {
			return
		}
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}

		body, err := decodeBody(r)
		if err != nil {
			writeAnthropicError(w, http.StatusBadRequest, err.Error())
			return
		}

		if boolValue(body["stream"], false) {
			if err := handleStreamingMessages(cfg, w, r, body); err != nil {
				writeAnthropicError(w, http.StatusBadGateway, err.Error())
			}
			return
		}
		response, err := handleMessages(cfg, r, body)
		if err != nil {
			writeAnthropicError(w, http.StatusBadGateway, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, response)
	})
	return mux
}

func loadRuntimeConfig() (*runtimeConfig, error) {
	configPath, err := resolveConfigPath()
	if err != nil {
		return nil, err
	}

	configBytes, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read proxy config: %w", err)
	}
	var fileCfg fileConfig
	if err := json.Unmarshal(configBytes, &fileCfg); err != nil {
		return nil, fmt.Errorf("parse proxy config: %w", err)
	}

	envFastMode := lookupBoolEnv("CLAUDE_PROXY_FAST_MODE")
	fileFastMode := firstConfiguredBool(fileCfg.FastMode, fileCfg.Fastmode)
	serviceTier := "priority"
	switch {
	case envFastMode != nil:
		serviceTier = serviceTierForFastMode(*envFastMode)
	case strings.TrimSpace(os.Getenv("CLAUDE_PROXY_SERVICE_TIER")) != "":
		serviceTier = strings.TrimSpace(os.Getenv("CLAUDE_PROXY_SERVICE_TIER"))
	case fileFastMode != nil:
		serviceTier = serviceTierForFastMode(*fileFastMode)
	}

	cfg := &runtimeConfig{
		Host:                firstNonEmpty(os.Getenv("CLAUDE_PROXY_HOST"), fileCfg.ListenHost, "127.0.0.1"),
		Port:                firstNonEmpty(os.Getenv("CLAUDE_PROXY_PORT"), fileCfg.ListenPort, "4000"),
		LocalAuthToken:      firstNonEmpty(os.Getenv("CLAUDE_PROXY_LOCAL_AUTH_TOKEN"), fileCfg.LocalAuthToken, ""),
		UpstreamBaseURL:     strings.TrimRight(firstNonEmpty(os.Getenv("CLAUDE_PROXY_UPSTREAM_BASE_URL"), fileCfg.UpstreamBaseURL, ""), "/"),
		UpstreamToken:       firstNonEmpty(os.Getenv("CLAUDE_PROXY_UPSTREAM_TOKEN"), fileCfg.UpstreamToken, ""),
		FastMode:            isFastModeServiceTier(serviceTier),
		ServiceTier:         serviceTier,
		ReasoningEffort:     configuredReasoningEffort(firstNonEmpty(os.Getenv("CLAUDE_PROXY_REASONING_EFFORT"), fileCfg.ReasoningEffort, "xhigh")),
		ContextWindowLength: intValue(os.Getenv("CLAUDE_PROXY_CONTEXT_WINDOW_LENGTH"), firstNonZero(fileCfg.ContextWindowLength, 1000000)),
		Model:               firstNonEmpty(os.Getenv("CLAUDE_PROXY_UPSTREAM_MODEL"), fileCfg.UpstreamModel, "gpt-5.4"),
		ConfigPath:          configPath,
	}

	if cfg.LocalAuthToken == "" {
		return nil, errors.New("missing local_auth_token in proxy config")
	}
	if cfg.UpstreamBaseURL == "" || cfg.UpstreamToken == "" {
		return nil, errors.New("missing upstream_base_url or upstream_token in proxy config")
	}
	if err := initializePersistentState(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

func ensureGeneratedClaudeSettings(cfg *runtimeConfig) error {
	settingsPath := filepath.Join(filepath.Dir(cfg.ConfigPath), "claude-settings.json")
	contextWindowLength := firstNonZero(cfg.ContextWindowLength, 1000000)
	settings := map[string]any{
		"model":           claudeSettingsModel(contextWindowLength),
		"availableModels": []string{"opus"},
		"effortLevel":     configuredReasoningEffort(cfg.ReasoningEffort),
		"env": map[string]string{
			"ANTHROPIC_BASE_URL":              fmt.Sprintf("http://%s:%s", cfg.Host, cfg.Port),
			"ANTHROPIC_AUTH_TOKEN":            cfg.LocalAuthToken,
			"ANTHROPIC_DEFAULT_SONNET_MODEL":  cfg.Model,
			"ANTHROPIC_DEFAULT_OPUS_MODEL":    cfg.Model,
			"ANTHROPIC_DEFAULT_HAIKU_MODEL":   cfg.Model,
			"CLAUDE_CODE_SUBAGENT_MODEL":      cfg.Model,
			"CLAUDE_CODE_EFFORT_LEVEL":        claudeCodeEffortLevel(cfg.ReasoningEffort),
			"CLAUDE_CODE_AUTO_COMPACT_WINDOW": strconv.Itoa(contextWindowLength),
		},
	}
	if contextWindowLength < 1000000 {
		settings["env"].(map[string]string)["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal generated claude settings: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(settingsPath, data, 0o644); err != nil {
		return fmt.Errorf("write generated claude settings: %w", err)
	}
	return nil
}

func resolveConfigPath() (string, error) {
	candidates := make([]string, 0)
	if configured := strings.TrimSpace(os.Getenv("CLAUDE_PROXY_CONFIG")); configured != "" {
		candidates = append(candidates, configured)
	}
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		candidates = append(candidates,
			filepath.Join(exeDir, "config.local.json"),
			filepath.Join(exeDir, "config.json"),
		)
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(cwd, "config.local.json"),
			filepath.Join(cwd, "config.json"),
		)
	}

	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, ok := seen[candidate]; ok {
			continue
		}
		seen[candidate] = struct{}{}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}

	return "", errors.New("could not find config.local.json or config.json; set CLAUDE_PROXY_CONFIG or place a config next to the binary")
}

func initializePersistentState(cfg *runtimeConfig) error {
	statePath := resolveStatePath(cfg.ConfigPath)
	state, err := readPersistedGatewayState(statePath)
	switch {
	case err == nil:
	case errors.Is(err, os.ErrNotExist):
		state = &persistedGatewayState{}
	default:
		return err
	}

	if state.CacheID == "" {
		state.CacheID = randomHex(8)
	}

	sessions.Replace(sessionStateMap(state.Sessions))
	cfg.CacheID = state.CacheID
	persistedState.Configure(statePath, state.CacheID)
	return persistedState.Save(sessions)
}

func resolveStatePath(configPath string) string {
	if configured := strings.TrimSpace(os.Getenv("CLAUDE_PROXY_SESSION_STATE_PATH")); configured != "" {
		return configured
	}
	return filepath.Join(filepath.Dir(configPath), "session-state.json")
}

func readPersistedGatewayState(path string) (*persistedGatewayState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var state persistedGatewayState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("parse session state: %w", err)
	}
	return &state, nil
}

func (p *statePersistence) Configure(path, cacheID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.path = path
	p.cacheID = cacheID
}

func (p *statePersistence) Save(store *sessionStore) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.path == "" || p.cacheID == "" {
		return nil
	}

	state := persistedGatewayState{
		CacheID:  p.cacheID,
		Sessions: pruneSessionStates(store.Snapshot(), p.maxAge, p.maxEntries),
	}
	return writeJSONFileAtomic(p.path, state)
}

func writeJSONFileAtomic(path string, payload any) error {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	tempPath := path + ".tmp"
	if err := os.WriteFile(tempPath, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tempPath, path)
}

func sessionStateMap(values []sessionState) map[string]sessionState {
	out := make(map[string]sessionState, len(values))
	for _, value := range values {
		if value.SessionID == "" || value.Model == "" || value.ResponseID == "" {
			continue
		}
		stateKey := strings.TrimSpace(value.StateKey)
		switch {
		case stateKey != "":
		case value.Synthetic:
			stateKey = value.SessionID
		default:
			stateKey = buildSessionStateKey(value.SessionID, value.PromptCacheKey)
		}
		if stateKey == "" {
			continue
		}
		value.StateKey = stateKey
		out[stateKey] = cloneSessionState(value)
	}
	return out
}

func pruneSessionStates(snapshot map[string]sessionState, maxAge time.Duration, maxEntries int) []sessionState {
	now := time.Now()
	states := make([]sessionState, 0, len(snapshot))
	for _, value := range snapshot {
		if value.UpdatedAt.IsZero() {
			value.UpdatedAt = now
		}
		if maxAge > 0 && now.Sub(value.UpdatedAt) > maxAge {
			continue
		}
		states = append(states, cloneSessionState(value))
	}

	sort.Slice(states, func(i, j int) bool {
		return states[i].UpdatedAt.After(states[j].UpdatedAt)
	})
	if maxEntries > 0 && len(states) > maxEntries {
		states = states[:maxEntries]
	}
	return states
}

func cloneSessionState(value sessionState) sessionState {
	value.MessageHashes = cloneStrings(value.MessageHashes)
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func firstNonZero(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstConfiguredBool(values ...*bool) *bool {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func lookupBoolEnv(key string) *bool {
	rawValue, ok := os.LookupEnv(key)
	if !ok {
		return nil
	}
	value, ok := parseFlexibleBool(rawValue)
	if !ok {
		return nil
	}
	return &value
}

func parseFlexibleBool(value string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "on", "yes":
		return true, true
	case "0", "false", "off", "no":
		return false, true
	default:
		return false, false
	}
}

func authenticate(cfg *runtimeConfig, w http.ResponseWriter, r *http.Request) bool {
	token := strings.TrimSpace(r.Header.Get("x-api-key"))
	if token == "" {
		auth := strings.TrimSpace(r.Header.Get("Authorization"))
		if strings.HasPrefix(auth, "Bearer ") {
			token = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if token != cfg.LocalAuthToken {
		writeAnthropicError(w, http.StatusUnauthorized, "Invalid API key")
		return false
	}
	return true
}

func decodeBody(r *http.Request) (map[string]any, error) {
	defer r.Body.Close()
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if len(bodyBytes) == 0 {
		return map[string]any{}, nil
	}
	var body map[string]any
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		return nil, err
	}
	return body, nil
}

func handleMessages(cfg *runtimeConfig, r *http.Request, body map[string]any) (map[string]any, error) {
	built, err := buildUpstreamRequest(cfg, r, body)
	if err != nil {
		return nil, err
	}
	requestedModel := stringValue(body["model"], cfg.Model)
	startedAt := time.Now()
	logTurnStart("sync", requestedModel, built)

	completed, err := sendUpstream(r.Context(), cfg, built)
	if err != nil {
		logTurnDone("sync", requestedModel, built, startedAt, 0, false, nil, err)
		return nil, err
	}

	storeSessionState(built, completed)

	translated := translateToAnthropic(completed, requestedModel)
	if built.CachedInputTokens > 0 {
		if usage, ok := translated["usage"].(map[string]any); ok {
			if intValue(usage["cache_read_input_tokens"], 0) == 0 {
				usage["cache_read_input_tokens"] = built.CachedInputTokens
			}
		}
	}
	logTurnDone("sync", requestedModel, built, startedAt, 0, false, completed, nil)
	return translated, nil
}

func handleStreamingMessages(cfg *runtimeConfig, w http.ResponseWriter, r *http.Request, body map[string]any) error {
	built, err := buildUpstreamRequest(cfg, r, body)
	if err != nil {
		return err
	}

	requestedModel := stringValue(body["model"], cfg.Model)
	startedAt := time.Now()
	logTurnStart("stream", requestedModel, built)
	completed, started, ttft, hasTTFT, err := streamUpstreamToAnthropic(r.Context(), cfg, built, requestedModel, startedAt, w)
	if err != nil {
		logTurnDone("stream", requestedModel, built, startedAt, ttft, hasTTFT, completed, err)
		if started {
			log.Printf("streaming response ended with error: %v", err)
			return nil
		}
		return err
	}

	storeSessionState(built, completed)
	logTurnDone("stream", requestedModel, built, startedAt, ttft, hasTTFT, completed, nil)
	return nil
}

type upstreamRequest struct {
	Primary               map[string]any
	Fallback              map[string]any
	UpstreamModel         string
	SessionID             string
	StateKey              string
	SyntheticSession      bool
	PromptCacheKey        string
	OriginalMessageHashes []string
	TotalMessageTokens    int
	MessageCount          int
	AppendedMessages      int
	PreviousResponseID    string
	ReuseMode             string
	EstimatedInput        int
	CachedInputTokens     int
}

func buildUpstreamRequest(cfg *runtimeConfig, r *http.Request, body map[string]any) (*upstreamRequest, error) {
	systemText := normalizeSystem(body["system"])
	tools := mapTools(body["tools"])
	textConfig := mapTextConfig(body["output_config"])
	promptCacheKey := buildPromptCacheKey(cfg.CacheID, cfg.Model, systemText, tools, textConfig)
	messages := anySlice(body["messages"])
	messageHashes, totalMessageTokens := hashMessages(messages)
	sessionID, stateKey, previous, hasPrevious, syntheticSession := resolveSessionState(r, body, promptCacheKey, messageHashes)
	requestMessages := messages
	cachedInputTokens := 0
	reuseMode := "full_history"

	if hasPrevious && previous.PromptCacheKey == promptCacheKey {
		cachedInputTokens = previous.TotalMessageTokens
		reuseMode = "full_history+cache"
	}

	input := convertMessages(requestMessages)
	instructions := systemText
	reasoning := map[string]any{
		"effort": configuredReasoningEffort(cfg.ReasoningEffort),
	}

	primary := map[string]any{
		"model":            cfg.Model,
		"input":            input,
		"reasoning":        reasoning,
		// The adapter always consumes the upstream Responses API as SSE.
		// Even for Claude's sync mode, we collect the SSE stream and return a
		// final JSON message after response.completed arrives.
		"stream":           true,
		"prompt_cache_key": promptCacheKey,
	}
	if cfg.ServiceTier != "" {
		primary["service_tier"] = cfg.ServiceTier
	}
	if instructions != "" {
		primary["instructions"] = instructions
	}
	if _, ok := body["max_tokens"]; ok {
		primary["max_output_tokens"] = intValue(body["max_tokens"], 0)
	}
	if temperature, ok := body["temperature"].(float64); ok {
		primary["temperature"] = temperature
	}
	if topP, ok := body["top_p"].(float64); ok {
		primary["top_p"] = topP
	}
	if textConfig != nil {
		primary["text"] = textConfig
	}
	if len(tools) > 0 {
		primary["tools"] = tools
	}
	if _, ok := body["tool_choice"]; ok {
		if toolChoice, ok := mapToolChoice(body["tool_choice"], tools); ok {
			primary["tool_choice"] = toolChoice
		}
	}
	fallback := cloneMap(primary)
	delete(fallback, "service_tier")

	return &upstreamRequest{
		Primary:               primary,
		Fallback:              fallback,
		UpstreamModel:         cfg.Model,
		SessionID:             sessionID,
		StateKey:              stateKey,
		SyntheticSession:      syntheticSession,
		PromptCacheKey:        promptCacheKey,
		OriginalMessageHashes: cloneStrings(messageHashes),
		TotalMessageTokens:    totalMessageTokens,
		MessageCount:          len(messages),
		AppendedMessages:      len(requestMessages),
		PreviousResponseID:    "",
		ReuseMode:             reuseMode,
		EstimatedInput:        estimateStaticInputTokens(systemText, tools, textConfig) + totalMessageTokens,
		CachedInputTokens:     cachedInputTokens,
	}, nil
}

func sendUpstream(ctx context.Context, cfg *runtimeConfig, req *upstreamRequest) (map[string]any, error) {
	turnID := fmt.Sprintf("%d", time.Now().UnixNano())
	headers := http.Header{
		"Authorization":         []string{"Bearer " + cfg.UpstreamToken},
		"content-type":          []string{"application/json"},
		"accept":                []string{"text/event-stream"},
		"user-agent":            []string{"codex_exec/0.120.0 (Mac OS 26.4.1; arm64) Apple_Terminal/470 (codex_exec; 0.120.0)"},
		"originator":            []string{"codex_exec"},
		"x-client-request-id":   []string{req.SessionID},
		"session_id":            []string{req.SessionID},
		"x-codex-window-id":     []string{req.SessionID + ":0"},
		"x-codex-turn-metadata": []string{fmt.Sprintf(`{"session_id":"%s","turn_id":"%s","sandbox":"seatbelt"}`, req.SessionID, turnID)},
	}

	responsesURL := upstreamResponsesURL(cfg.UpstreamBaseURL)
	tryList := []struct {
		url     string
		payload map[string]any
	}{
		{url: responsesURL, payload: req.Primary},
		{url: responsesURL, payload: req.Fallback},
	}

	var lastErr error
	for _, attempt := range tryList {
		result, err := performResponsesRequest(ctx, attempt.url, attempt.payload, headers)
		if err == nil {
			return result, nil
		}
		lastErr = err
	}

	return nil, lastErr
}

func upstreamResponsesURL(base string) string {
	base = strings.TrimRight(base, "/")
	base = strings.TrimSuffix(base, "/v1")
	return base + "/v1/responses"
}

func performResponsesRequest(ctx context.Context, url string, payload map[string]any, headers http.Header) (map[string]any, error) {
	response, err := openResponsesRequest(ctx, url, payload, headers)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	return collectResponses(response.Body)
}

func openResponsesRequest(ctx context.Context, url string, payload map[string]any, headers http.Header) (*http.Response, error) {
	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	request.Header = headers.Clone()

	response, err := responsesHTTPClient.Do(request)
	if err != nil {
		return nil, err
	}

	if response.StatusCode >= 400 {
		raw, _ := io.ReadAll(response.Body)
		_ = response.Body.Close()
		return nil, fmt.Errorf("upstream %d: %s", response.StatusCode, string(raw))
	}

	return response, nil
}

func collectResponses(body io.Reader) (map[string]any, error) {
	var completed map[string]any
	outputByIndex := map[int]map[string]any{}
	if err := processSSEStream(body, func(event sseEvent) error {
		parsed, err := parseSSEJSON(event)
		if err != nil || parsed == nil {
			return err
		}

		switch stringValue(parsed["type"], "") {
		case "response.output_item.done":
			idx := intValue(parsed["output_index"], 0)
			if item, ok := parsed["item"].(map[string]any); ok {
				outputByIndex[idx] = item
			}
		case "response.completed":
			if responseMap, ok := parsed["response"].(map[string]any); ok {
				completed = responseMap
			}
		case "response.failed", "error":
			return upstreamStreamError(parsed)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	if completed == nil {
		return nil, errors.New("missing response.completed event")
	}

	output := anySlice(completed["output"])
	if len(output) == 0 && len(outputByIndex) > 0 {
		indices := make([]int, 0, len(outputByIndex))
		for idx := range outputByIndex {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		rebuilt := make([]any, 0, len(indices))
		for _, idx := range indices {
			rebuilt = append(rebuilt, outputByIndex[idx])
		}
		completed["output"] = rebuilt
	}

	return completed, nil
}

func streamUpstreamToAnthropic(ctx context.Context, cfg *runtimeConfig, req *upstreamRequest, requestedModel string, requestStartedAt time.Time, w http.ResponseWriter) (map[string]any, bool, time.Duration, bool, error) {
	turnID := fmt.Sprintf("%d", time.Now().UnixNano())
	headers := http.Header{
		"Authorization":         []string{"Bearer " + cfg.UpstreamToken},
		"content-type":          []string{"application/json"},
		"accept":                []string{"text/event-stream"},
		"user-agent":            []string{"codex_exec/0.120.0 (Mac OS 26.4.1; arm64) Apple_Terminal/470 (codex_exec; 0.120.0)"},
		"originator":            []string{"codex_exec"},
		"x-client-request-id":   []string{req.SessionID},
		"session_id":            []string{req.SessionID},
		"x-codex-window-id":     []string{req.SessionID + ":0"},
		"x-codex-turn-metadata": []string{fmt.Sprintf(`{"session_id":"%s","turn_id":"%s","sandbox":"seatbelt"}`, req.SessionID, turnID)},
	}

	responsesURL := upstreamResponsesURL(cfg.UpstreamBaseURL)
	tryList := []struct {
		url     string
		payload map[string]any
	}{
		{url: responsesURL, payload: req.Primary},
		{url: responsesURL, payload: req.Fallback},
	}

	var lastErr error
	for _, attempt := range tryList {
		response, err := openResponsesRequest(ctx, attempt.url, attempt.payload, headers)
		if err != nil {
			lastErr = err
			continue
		}

		streamer, err := newAnthropicStreamWriter(w, requestedModel, req.EstimatedInput, req.CachedInputTokens, requestStartedAt)
		if err != nil {
			_ = response.Body.Close()
			return nil, false, 0, false, err
		}
		completed, err := translateResponsesStream(response.Body, streamer)
		_ = response.Body.Close()
		if err != nil {
			return nil, streamer.started, streamer.firstOutputDelay(), streamer.hasFirstOutput(), err
		}
		return completed, streamer.started, streamer.firstOutputDelay(), streamer.hasFirstOutput(), nil
	}

	return nil, false, 0, false, lastErr
}

func translateToAnthropic(response map[string]any, requestedModel string) map[string]any {
	content := make([]map[string]any, 0)
	stopReason := "end_turn"

	for _, item := range anySlice(response["output"]) {
		itemMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		switch itemMap["type"] {
		case "message":
			for _, part := range anySlice(itemMap["content"]) {
				partMap, ok := part.(map[string]any)
				if !ok {
					continue
				}
				if partMap["type"] == "output_text" {
					content = append(content, map[string]any{
						"type": "text",
						"text": stringValue(partMap["text"], ""),
					})
				}
			}
		case "function_call":
			input := map[string]any{}
			if rawArgs, ok := itemMap["arguments"].(string); ok && rawArgs != "" {
				_ = json.Unmarshal([]byte(rawArgs), &input)
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    normalizeToolCallID(stringValue(itemMap["call_id"], stringValue(itemMap["id"], ""))),
				"name":  normalizeToolName(stringValue(itemMap["name"], "")),
				"input": input,
			})
			stopReason = "tool_use"
		}
	}

	return map[string]any{
		"id":            stringValue(response["id"], ""),
		"type":          "message",
		"role":          "assistant",
		"model":         requestedModel,
		"stop_sequence": nil,
		"usage":         anthropicUsageMap(response, 0),
		"content":       content,
		"stop_reason":   stopReason,
	}
}

func writeAnthropicSSE(w http.ResponseWriter, response map[string]any) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusOK, response)
		return
	}

	w.Header().Set("content-type", "text/event-stream")
	w.Header().Set("cache-control", "no-cache")
	w.Header().Set("connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	send := func(eventType string, payload map[string]any) {
		_, _ = fmt.Fprintf(w, "event: %s\n", eventType)
		data, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	usage, _ := response["usage"].(map[string]any)
	send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            response["id"],
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         response["model"],
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         anthropicStreamUsage(usage),
		},
	})

	blocks := normalizeContentBlocks(response["content"])
	for index, block := range blocks {
		switch block["type"] {
		case "text":
			send("content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         index,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			send("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": index,
				"delta": map[string]any{"type": "text_delta", "text": stringValue(block["text"], "")},
			})
			send("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": index,
			})
		case "tool_use":
			toolID := normalizeToolCallID(stringValue(block["id"], ""))
			send("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": index,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    toolID,
					"name":  normalizeToolName(stringValue(block["name"], "")),
					"input": map[string]any{},
				},
			})
			send("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": index,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": marshalString(block["input"]),
				},
			})
			send("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": index,
			})
		}
	}

	messageDelta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stringValue(response["stop_reason"], "end_turn"),
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"input_tokens":  intValue(usage["input_tokens"], 0),
			"output_tokens": intValue(usage["output_tokens"], 0),
		},
	}
	if cached := intValue(usage["cache_read_input_tokens"], 0); cached > 0 {
		messageDelta["usage"].(map[string]any)["cache_read_input_tokens"] = cached
	}
	send("message_delta", messageDelta)
	send("message_stop", map[string]any{"type": "message_stop"})
}

func anthropicStreamUsage(usage map[string]any) map[string]any {
	streamUsage := map[string]any{
		"input_tokens":                intValue(usage["input_tokens"], 0),
		"output_tokens":               intValue(usage["output_tokens"], 0),
		"cache_creation_input_tokens": intValue(usage["cache_creation_input_tokens"], 0),
		"cache_read_input_tokens":     intValue(usage["cache_read_input_tokens"], 0),
	}
	if total := intValue(usage["total_tokens"], 0); total > 0 {
		streamUsage["total_tokens"] = total
	}
	return streamUsage
}

func normalizeContentBlocks(value any) []map[string]any {
	switch v := value.(type) {
	case []map[string]any:
		return v
	case []any:
		out := make([]map[string]any, 0, len(v))
		for _, item := range v {
			if block, ok := item.(map[string]any); ok {
				out = append(out, block)
			}
		}
		return out
	default:
		return nil
	}
}

type sseEvent struct {
	Event string
	Data  string
}

func processSSEStream(body io.Reader, handler func(event sseEvent) error) error {
	reader := bufio.NewReader(body)
	var eventType string
	var dataLines []string

	flushEvent := func() error {
		if eventType == "" && len(dataLines) == 0 {
			return nil
		}
		event := sseEvent{
			Event: eventType,
			Data:  strings.Join(dataLines, "\n"),
		}
		eventType = ""
		dataLines = nil
		return handler(event)
	}

	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			switch {
			case line == "":
				if err := flushEvent(); err != nil {
					return err
				}
			case strings.HasPrefix(line, "event:"):
				eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			case strings.HasPrefix(line, "data:"):
				dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			}
		}
		if err != nil {
			if err == io.EOF {
				if err := flushEvent(); err != nil {
					return err
				}
				return nil
			}
			return err
		}
	}
}

func parseSSEJSON(event sseEvent) (map[string]any, error) {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil, nil
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(event.Data), &parsed); err != nil {
		return nil, fmt.Errorf("decode upstream event: %w", err)
	}
	return parsed, nil
}

type anthropicStreamWriter struct {
	w                 http.ResponseWriter
	flusher           http.Flusher
	requestedModel    string
	estimatedInput    int
	cachedInputTokens int
	requestStartedAt  time.Time
	firstOutputAt     time.Time
	started           bool
	messageID         string
	nextIndex         int
	blockIndexes      map[string]int
	openBlocks        map[string]bool
	toolNames         map[string]string
	toolCallIDs       map[string]string
	toolArgumentsSent map[string]bool
	stopReason        string
}

func newAnthropicStreamWriter(w http.ResponseWriter, requestedModel string, estimatedInput int, cachedInputTokens int, requestStartedAt time.Time) (*anthropicStreamWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("response writer does not support streaming")
	}
	return &anthropicStreamWriter{
		w:                 w,
		flusher:           flusher,
		requestedModel:    requestedModel,
		estimatedInput:    estimatedInput,
		cachedInputTokens: cachedInputTokens,
		requestStartedAt:  requestStartedAt,
		blockIndexes:      map[string]int{},
		openBlocks:        map[string]bool{},
		toolNames:         map[string]string{},
		toolCallIDs:       map[string]string{},
		toolArgumentsSent: map[string]bool{},
		stopReason:        "end_turn",
	}, nil
}

func (s *anthropicStreamWriter) send(eventType string, payload map[string]any) {
	_, _ = fmt.Fprintf(s.w, "event: %s\n", eventType)
	data, _ := json.Marshal(payload)
	_, _ = fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.flusher.Flush()
}

func (s *anthropicStreamWriter) markFirstOutput() {
	if !s.firstOutputAt.IsZero() {
		return
	}
	s.firstOutputAt = time.Now()
}

func (s *anthropicStreamWriter) hasFirstOutput() bool {
	return !s.firstOutputAt.IsZero()
}

func (s *anthropicStreamWriter) firstOutputDelay() time.Duration {
	if s.firstOutputAt.IsZero() || s.requestStartedAt.IsZero() {
		return 0
	}
	return s.firstOutputAt.Sub(s.requestStartedAt).Round(time.Millisecond)
}

func (s *anthropicStreamWriter) startMessage(messageID string) {
	if s.started {
		if s.messageID == "" && messageID != "" {
			s.messageID = messageID
		}
		return
	}

	s.messageID = firstNonEmpty(messageID, s.messageID)
	s.w.Header().Set("content-type", "text/event-stream")
	s.w.Header().Set("cache-control", "no-cache")
	s.w.Header().Set("connection", "keep-alive")
	s.w.WriteHeader(http.StatusOK)
	s.started = true
	startUsage := map[string]any{
		"input_tokens":  s.estimatedInput,
		"output_tokens": 0,
	}
	if s.cachedInputTokens > 0 {
		startUsage["cache_read_input_tokens"] = s.cachedInputTokens
	}
	if s.estimatedInput > 0 {
		startUsage["total_tokens"] = s.estimatedInput
	}
	s.send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            s.messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         s.requestedModel,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         startUsage,
		},
	})
}

func (s *anthropicStreamWriter) ensureTextBlock(key, messageID string) int {
	s.startMessage(messageID)
	if index, ok := s.blockIndexes[key]; ok {
		return index
	}
	index := s.nextIndex
	s.nextIndex++
	s.blockIndexes[key] = index
	s.openBlocks[key] = true
	s.markFirstOutput()
	s.send("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         index,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	return index
}

func (s *anthropicStreamWriter) appendTextDelta(key, messageID, delta string) {
	index := s.ensureTextBlock(key, messageID)
	s.send("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]any{"type": "text_delta", "text": delta},
	})
}

func (s *anthropicStreamWriter) ensureToolBlock(key, messageID, callID, name string) int {
	s.startMessage(messageID)
	if callID != "" {
		s.toolCallIDs[key] = normalizeToolCallID(callID)
	}
	if name != "" {
		s.toolNames[key] = normalizeToolName(name)
	}
	if index, ok := s.blockIndexes[key]; ok {
		return index
	}
	index := s.nextIndex
	s.nextIndex++
	s.blockIndexes[key] = index
	s.openBlocks[key] = true
	s.stopReason = "tool_use"
	s.markFirstOutput()
	s.send("content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": index,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    firstNonEmpty(s.toolCallIDs[key], key),
			"name":  s.toolNames[key],
			"input": map[string]any{},
		},
	})
	return index
}

func (s *anthropicStreamWriter) appendToolDelta(key, messageID, callID, name, delta string) {
	if delta == "" {
		return
	}
	index := s.ensureToolBlock(key, messageID, callID, name)
	s.toolArgumentsSent[key] = true
	s.send("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]any{
			"type":         "input_json_delta",
			"partial_json": delta,
		},
	})
}

func (s *anthropicStreamWriter) closeBlock(key string) {
	if !s.openBlocks[key] {
		return
	}
	s.openBlocks[key] = false
	s.send("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": s.blockIndexes[key],
	})
}

func (s *anthropicStreamWriter) finish(response map[string]any) {
	s.startMessage(stringValue(response["id"], ""))
	for key := range s.openBlocks {
		s.closeBlock(key)
	}

	usage := anthropicUsageMap(response, s.cachedInputTokens)
	messageDelta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   s.stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"input_tokens":  intValue(usage["input_tokens"], 0),
			"output_tokens": intValue(usage["output_tokens"], 0),
		},
	}
	if cached := intValue(usage["cache_read_input_tokens"], 0); cached > 0 {
		messageDelta["usage"].(map[string]any)["cache_read_input_tokens"] = cached
	}
	if total := intValue(usage["total_tokens"], 0); total > 0 {
		messageDelta["usage"].(map[string]any)["total_tokens"] = total
	}
	s.send("message_delta", messageDelta)
	s.send("message_stop", map[string]any{"type": "message_stop"})
}

func translateResponsesStream(body io.Reader, streamer *anthropicStreamWriter) (map[string]any, error) {
	var completed map[string]any
	if err := processSSEStream(body, func(event sseEvent) error {
		parsed, err := parseSSEJSON(event)
		if err != nil || parsed == nil {
			return err
		}

		responseID := firstNonEmpty(
			stringValue(parsed["response_id"], ""),
			nestedString(parsed, "response", "id"),
		)
		outputIndex := intValue(parsed["output_index"], 0)

		switch stringValue(parsed["type"], "") {
		case "response.created":
			streamer.startMessage(responseID)
		case "response.output_text.delta":
			key := fmt.Sprintf("text:%d:%d", outputIndex, intValue(parsed["content_index"], 0))
			streamer.appendTextDelta(key, responseID, stringValue(parsed["delta"], ""))
		case "response.output_text.done":
			key := fmt.Sprintf("text:%d:%d", outputIndex, intValue(parsed["content_index"], 0))
			streamer.closeBlock(key)
		case "response.refusal.delta":
			key := fmt.Sprintf("refusal:%d:%d", outputIndex, intValue(parsed["content_index"], 0))
			streamer.appendTextDelta(key, responseID, stringValue(parsed["delta"], ""))
		case "response.refusal.done":
			key := fmt.Sprintf("refusal:%d:%d", outputIndex, intValue(parsed["content_index"], 0))
			streamer.closeBlock(key)
		case "response.output_item.added":
			item, _ := parsed["item"].(map[string]any)
			if stringValue(item["type"], "") == "function_call" {
				key := firstNonEmpty(stringValue(item["id"], ""), fmt.Sprintf("tool:%d", outputIndex))
				streamer.ensureToolBlock(key, responseID, stringValue(item["call_id"], ""), stringValue(item["name"], ""))
			}
		case "response.function_call_arguments.delta":
			key := firstNonEmpty(stringValue(parsed["item_id"], ""), fmt.Sprintf("tool:%d", outputIndex))
			streamer.appendToolDelta(key, responseID, stringValue(parsed["call_id"], ""), streamer.toolNames[key], stringValue(parsed["delta"], ""))
		case "response.function_call_arguments.done":
			key := firstNonEmpty(stringValue(parsed["item_id"], ""), fmt.Sprintf("tool:%d", outputIndex))
			if !streamer.toolArgumentsSent[key] {
				streamer.appendToolDelta(key, responseID, stringValue(parsed["call_id"], ""), stringValue(parsed["name"], streamer.toolNames[key]), stringValue(parsed["arguments"], ""))
			}
			streamer.closeBlock(key)
		case "response.output_item.done":
			item, _ := parsed["item"].(map[string]any)
			if stringValue(item["type"], "") == "function_call" {
				key := firstNonEmpty(stringValue(item["id"], ""), fmt.Sprintf("tool:%d", outputIndex))
				if arguments := stringValue(item["arguments"], ""); arguments != "" && !streamer.toolArgumentsSent[key] {
					streamer.appendToolDelta(key, responseID, stringValue(item["call_id"], ""), stringValue(item["name"], streamer.toolNames[key]), arguments)
				}
				streamer.closeBlock(key)
			}
		case "response.completed":
			if responseMap, ok := parsed["response"].(map[string]any); ok {
				completed = responseMap
				streamer.finish(responseMap)
			}
		case "response.failed", "error":
			return upstreamStreamError(parsed)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	if completed == nil {
		return nil, errors.New("missing response.completed event")
	}
	return completed, nil
}

func extractExplicitSessionID(r *http.Request, body map[string]any) string {
	if sessionID := strings.TrimSpace(r.Header.Get("x-claude-code-session-id")); sessionID != "" {
		return sessionID
	}
	if metadata, ok := body["metadata"].(map[string]any); ok {
		if sessionID := stringValue(metadata["session_id"], ""); sessionID != "" {
			return sessionID
		}
		if sessionID := stringValue(metadata["conversation_id"], ""); sessionID != "" {
			return sessionID
		}
		if rawUser, ok := metadata["user_id"].(string); ok && rawUser != "" {
			var parsed map[string]any
			if json.Unmarshal([]byte(rawUser), &parsed) == nil {
				if sessionID := stringValue(parsed["session_id"], ""); sessionID != "" {
					return sessionID
				}
				if sessionID := stringValue(parsed["conversation_id"], ""); sessionID != "" {
					return sessionID
				}
			}
		}
	}
	return ""
}

func normalizeSystem(system any) string {
	switch value := system.(type) {
	case string:
		if strings.HasPrefix(value, "x-anthropic-billing-header:") {
			return ""
		}
		return value
	case []any:
		parts := make([]string, 0, len(value))
		for _, blockValue := range value {
			block, ok := blockValue.(map[string]any)
			if !ok || stringValue(block["type"], "") != "text" {
				continue
			}
			text := stringValue(block["text"], "")
			if strings.HasPrefix(text, "x-anthropic-billing-header:") {
				continue
			}
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func mapTools(rawTools any) []any {
	tools := anySlice(rawTools)
	out := make([]any, 0, len(tools))
	for _, toolValue := range tools {
		tool, ok := toolValue.(map[string]any)
		if !ok {
			continue
		}
		if strings.HasPrefix(stringValue(tool["type"], ""), "web_search") {
			out = append(out, map[string]any{"type": "web_search_preview"})
			continue
		}
		name := stringValue(tool["name"], "")
		if name == "" {
			continue
		}
		mapped := map[string]any{
			"type": "function",
			"name": name,
		}
		if description := stringValue(tool["description"], ""); description != "" {
			mapped["description"] = description
		}
		if parameters, ok := tool["input_schema"]; ok {
			mapped["parameters"] = parameters
		}
		out = append(out, mapped)
	}
	return out
}

func convertMessages(messages []any) []any {
	out := make([]any, 0)
	for _, messageValue := range messages {
		message, ok := messageValue.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, convertMessage(message)...)
	}
	return out
}

func convertMessage(message map[string]any) []any {
	role := stringValue(message["role"], "user")
	content := message["content"]
	out := make([]any, 0)

	switch value := content.(type) {
	case string:
		if value == "" {
			return nil
		}
		return []any{map[string]any{
			"role": role,
			"content": []any{map[string]any{
				"type": messageContentType(role),
				"text": value,
			}},
		}}
	case []any:
		messageContent := make([]any, 0)
		flushMessage := func() {
			if len(messageContent) == 0 {
				return
			}
			out = append(out, map[string]any{
				"role":    role,
				"content": append([]any(nil), messageContent...),
			})
			messageContent = messageContent[:0]
		}
		for _, blockValue := range value {
			block, ok := blockValue.(map[string]any)
			if !ok {
				continue
			}
			switch stringValue(block["type"], "") {
			case "text":
				text := stringValue(block["text"], "")
				if text == "" {
					continue
				}
				messageContent = append(messageContent, map[string]any{
					"type": messageContentType(role),
					"text": text,
				})
			case "image":
				source, ok := block["source"].(map[string]any)
				if !ok {
					continue
				}
				data := stringValue(source["data"], "")
				if data == "" {
					continue
				}
				messageContent = append(messageContent, map[string]any{
					"type":      "input_image",
					"image_url": fmt.Sprintf("data:%s;base64,%s", stringValue(source["media_type"], "image/png"), data),
				})
			case "tool_use":
				flushMessage()
				out = append(out, map[string]any{
					"type":      "function_call",
					"call_id":   normalizeToolCallID(stringValue(block["id"], "")),
					"name":      normalizeToolName(stringValue(block["name"], "")),
					"arguments": marshalString(block["input"]),
				})
			case "tool_result":
				flushMessage()
				output := ""
				switch blockContent := block["content"].(type) {
				case string:
					output = blockContent
				default:
					output = marshalString(block["content"])
				}
				out = append(out, map[string]any{
					"type":    "function_call_output",
					"call_id": normalizeToolCallID(stringValue(block["tool_use_id"], "")),
					"output":  output,
				})
			case "thinking":
				// Match cc-switch custom openai_responses transform and drop thinking blocks.
			}
		}
		flushMessage()
	default:
		return []any{map[string]any{"role": role}}
	}

	return out
}

func messageContentType(role string) string {
	if role == "assistant" {
		return "output_text"
	}
	return "input_text"
}

func mapToolChoice(toolChoice any, tools []any) (any, bool) {
	toolNames := mappedToolNames(tools)

	switch choice := toolChoice.(type) {
	case string:
		return mapStringToolChoice(choice, toolNames)
	case map[string]any:
		return mapObjectToolChoice(choice, toolNames)
	default:
		if len(toolNames) == 0 {
			return nil, false
		}
		return "auto", true
	}
}

func mapStringToolChoice(choice string, toolNames map[string]string) (any, bool) {
	normalized := strings.ToLower(strings.TrimSpace(choice))
	switch normalized {
	case "":
		if len(toolNames) == 0 {
			return nil, false
		}
		return "auto", true
	case "none":
		return "none", true
	case "auto":
		if len(toolNames) == 0 {
			return nil, false
		}
		return "auto", true
	case "any", "required", "function", "tool":
		if len(toolNames) == 0 {
			return nil, false
		}
		return "required", true
	default:
		if name, ok := resolveMappedToolName(choice, toolNames); ok {
			return map[string]any{"type": "function", "name": name}, true
		}
		if len(toolNames) == 0 {
			return nil, false
		}
		return "auto", true
	}
}

func mapObjectToolChoice(choice map[string]any, toolNames map[string]string) (any, bool) {
	choiceType := strings.ToLower(strings.TrimSpace(stringValue(choice["type"], "")))
	switch choiceType {
	case "", "function":
		name := stringValue(choice["name"], "")
		if function, ok := choice["function"].(map[string]any); ok && name == "" {
			name = stringValue(function["name"], "")
		}
		if name != "" {
			if resolved, ok := resolveMappedToolName(name, toolNames); ok {
				return map[string]any{"type": "function", "name": resolved}, true
			}
		}
	case "tool":
		name := stringValue(choice["name"], "")
		if resolved, ok := resolveMappedToolName(name, toolNames); ok {
			return map[string]any{"type": "function", "name": resolved}, true
		}
	case "any", "required":
		if len(toolNames) == 0 {
			return nil, false
		}
		return "required", true
	case "auto":
		if len(toolNames) == 0 {
			return nil, false
		}
		return "auto", true
	case "none":
		return "none", true
	}

	if len(toolNames) == 0 {
		return nil, false
	}
	return "auto", true
}

func mappedToolNames(tools []any) map[string]string {
	names := make(map[string]string, len(tools))
	for _, toolValue := range tools {
		tool, ok := toolValue.(map[string]any)
		if !ok {
			continue
		}
		name := stringValue(tool["name"], "")
		if name == "" {
			continue
		}
		names[strings.ToLower(name)] = name
	}
	return names
}

func resolveMappedToolName(name string, toolNames map[string]string) (string, bool) {
	if len(toolNames) == 0 {
		return "", false
	}
	resolved, ok := toolNames[strings.ToLower(strings.TrimSpace(name))]
	return resolved, ok
}

func mapTextConfig(outputConfig any) map[string]any {
	cfg, ok := outputConfig.(map[string]any)
	if !ok {
		return nil
	}
	format, ok := cfg["format"].(map[string]any)
	if !ok {
		return nil
	}
	if stringValue(format["type"], "") != "json_schema" {
		return nil
	}
	schema := format["schema"]
	if schema == nil {
		return nil
	}
	return map[string]any{
		"verbosity": "medium",
		"format": map[string]any{
			"type":   "json_schema",
			"name":   "structured_output",
			"schema": schema,
			"strict": true,
		},
	}
}

func normalizeReasoningEffort(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "minimal":
		return "low"
	case "max":
		return "xhigh"
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func configuredReasoningEffort(value string) string {
	if normalized := normalizeReasoningEffort(value); normalized != "" {
		return normalized
	}
	return "xhigh"
}

func claudeCodeEffortLevel(value string) string {
	switch configuredReasoningEffort(value) {
	case "xhigh":
		return "max"
	default:
		return configuredReasoningEffort(value)
	}
}

func serviceTierForFastMode(fastMode bool) string {
	if fastMode {
		return "priority"
	}
	return ""
}

func isFastModeServiceTier(value string) bool {
	return strings.EqualFold(strings.TrimSpace(value), "priority")
}

func claudeSettingsModel(contextWindowLength int) string {
	if contextWindowLength >= 1000000 {
		return "opus[1m]"
	}
	return "opus"
}

func buildPromptCacheKey(cacheID, model, instructions string, tools []any, textConfig map[string]any) string {
	if strings.TrimSpace(cacheID) != "" {
		return cacheID
	}
	sum := sha256Hex(stableJSON(map[string]any{
		"cache_id":     cacheID,
		"model":        model,
		"instructions": instructions,
		"tools":        tools,
		"text":         textConfig,
	}))
	return "claude-gpt54-" + sum[:32]
}

func normalizeToolCallID(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) <= 64 {
		return value
	}
	return "call_" + sha256Hex(value)[:59]
}

func normalizeToolName(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "unknown_tool"
	}
	return value
}

func storeSessionState(built *upstreamRequest, completed map[string]any) {
	if built == nil || built.StateKey == "" || len(built.OriginalMessageHashes) == 0 {
		return
	}
	if responseID, _ := completed["id"].(string); responseID != "" {
		sessions.Set(built.StateKey, sessionState{
			StateKey:           built.StateKey,
			SessionID:          built.SessionID,
			Model:              built.UpstreamModel,
			PromptCacheKey:     built.PromptCacheKey,
			ResponseID:         responseID,
			MessageHashes:      cloneStrings(built.OriginalMessageHashes),
			TotalMessageTokens: built.TotalMessageTokens,
			Synthetic:          built.SyntheticSession,
		})
		if err := persistedState.Save(sessions); err != nil {
			log.Printf("persist session state failed: %v", err)
		}
	}
}

func logTurnStart(mode, requestedModel string, req *upstreamRequest) {
	if req == nil {
		return
	}
	log.Printf(
		"turn_start mode=%s model=%s session=%s synthetic=%t prompt_cache=%s reuse=%s messages=%d appended=%d prev_response=%s estimated_input=%d cached_input=%d",
		mode,
		requestedModel,
		shortID(req.SessionID),
		req.SyntheticSession,
		shortID(req.PromptCacheKey),
		req.ReuseMode,
		req.MessageCount,
		req.AppendedMessages,
		shortID(req.PreviousResponseID),
		req.EstimatedInput,
		req.CachedInputTokens,
	)
}

func logTurnDone(mode, requestedModel string, req *upstreamRequest, startedAt time.Time, ttft time.Duration, hasTTFT bool, completed map[string]any, err error) {
	if req == nil {
		return
	}

	duration := time.Since(startedAt).Round(time.Millisecond)
	ttftLabel := "-"
	if hasTTFT {
		ttftLabel = ttft.String()
	}
	usage := anthropicUsageMap(completed, req.CachedInputTokens)
	inputTokens := intValue(usage["input_tokens"], 0)
	outputTokens := intValue(usage["output_tokens"], 0)
	totalTokens := intValue(usage["total_tokens"], 0)
	cacheReadTokens := intValue(usage["cache_read_input_tokens"], 0)
	if err != nil {
		log.Printf(
			"turn_done mode=%s model=%s session=%s response=%s ttft=%s duration=%s input_tokens=%d output_tokens=%d total_tokens=%d cache_read_input_tokens=%d err=%q",
			mode,
			requestedModel,
			shortID(req.SessionID),
			shortID(nestedString(completed, "id")),
			ttftLabel,
			duration,
			inputTokens,
			outputTokens,
			totalTokens,
			cacheReadTokens,
			err.Error(),
		)
		return
	}

	log.Printf(
		"turn_done mode=%s model=%s session=%s response=%s ttft=%s duration=%s input_tokens=%d output_tokens=%d total_tokens=%d cache_read_input_tokens=%d",
		mode,
		requestedModel,
		shortID(req.SessionID),
		shortID(nestedString(completed, "id")),
		ttftLabel,
		duration,
		inputTokens,
		outputTokens,
		totalTokens,
		cacheReadTokens,
	)
}

func writeJSON(w http.ResponseWriter, statusCode int, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(statusCode)
	_, _ = w.Write(data)
}

func writeAnthropicError(w http.ResponseWriter, statusCode int, message string) {
	writeJSON(w, statusCode, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "bad_request_error",
			"param":   nil,
			"code":    strconv.Itoa(statusCode),
		},
	})
}

func anthropicUsageMap(response map[string]any, cachedFallback int) map[string]any {
	usageMap, _ := response["usage"].(map[string]any)
	inputDetails, _ := usageMap["input_tokens_details"].(map[string]any)
	anthropicUsage := map[string]any{
		"input_tokens":  intValue(usageMap["input_tokens"], 0),
		"output_tokens": intValue(usageMap["output_tokens"], 0),
		"total_tokens":  intValue(usageMap["total_tokens"], 0),
	}
	if cached := intValue(inputDetails["cached_tokens"], 0); cached > 0 {
		anthropicUsage["cache_read_input_tokens"] = cached
	} else if cachedFallback > 0 {
		anthropicUsage["cache_read_input_tokens"] = cachedFallback
	}
	return anthropicUsage
}

func upstreamStreamError(event map[string]any) error {
	if responseMap, ok := event["response"].(map[string]any); ok {
		if errMap, ok := responseMap["error"].(map[string]any); ok {
			return errors.New(firstNonEmpty(stringValue(errMap["message"], ""), marshalString(errMap)))
		}
	}
	if errMap, ok := event["error"].(map[string]any); ok {
		return errors.New(firstNonEmpty(stringValue(errMap["message"], ""), marshalString(errMap)))
	}
	if message := stringValue(event["message"], ""); message != "" {
		return errors.New(message)
	}
	return fmt.Errorf("upstream stream failed: %s", stringValue(event["type"], "error"))
}

func approximateTokens(payload any) int {
	return len(stableJSON(payload)) / 4
}

func estimateStaticInputTokens(instructions string, tools []any, textConfig map[string]any) int {
	staticInput := map[string]any{}
	if instructions != "" {
		staticInput["instructions"] = instructions
	}
	if len(tools) > 0 {
		staticInput["tools"] = tools
	}
	if textConfig != nil {
		staticInput["text"] = textConfig
	}
	if len(staticInput) == 0 {
		return 0
	}
	return approximateTokens(staticInput)
}

func hashMessages(messages []any) ([]string, int) {
	hashes := make([]string, 0, len(messages))
	totalTokens := 0
	for _, message := range messages {
		rawSerialized := stableJSON(message)
		hashes = append(hashes, messageSemanticHash(message))
		totalTokens += len(rawSerialized) / 4
	}
	return hashes, totalTokens
}

func messageSemanticHash(message any) string {
	messageMap, ok := message.(map[string]any)
	if !ok {
		return sha256Hex(stableJSON(message))
	}
	return sha256Hex(stableJSON(convertMessage(messageMap)))
}

func anySlice(value any) []any {
	if slice, ok := value.([]any); ok {
		return slice
	}
	return nil
}

func intValue(value any, fallback int) int {
	switch v := value.(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	case json.Number:
		if i, err := v.Int64(); err == nil {
			return int(i)
		}
	case string:
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func boolValue(value any, fallback bool) bool {
	if v, ok := value.(bool); ok {
		return v
	}
	return fallback
}

func stringValue(value any, fallback string) string {
	if v, ok := value.(string); ok && v != "" {
		return v
	}
	return fallback
}

func nestedString(root map[string]any, path ...string) string {
	current := any(root)
	for _, key := range path {
		nextMap, ok := current.(map[string]any)
		if !ok {
			return ""
		}
		current = nextMap[key]
	}
	return stringValue(current, "")
}

func cloneMap(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func resolveSessionState(r *http.Request, body map[string]any, promptCacheKey string, messageHashes []string) (string, string, sessionState, bool, bool) {
	if sessionID := extractExplicitSessionID(r, body); sessionID != "" {
		stateKey := buildSessionStateKey(sessionID, promptCacheKey)
		previous, ok := sessions.Get(stateKey)
		return sessionID, stateKey, previous, ok, false
	}

	if stateKey, previous, ok := sessions.FindSyntheticPrefix(promptCacheKey, messageHashes); ok {
		return stateKey, stateKey, previous, true, true
	}

	sessionID := newSyntheticStateKey(promptCacheKey)
	stateKey := sessionID
	return sessionID, stateKey, sessionState{}, false, true
}

func buildSessionStateKey(sessionID, promptCacheKey string) string {
	return sha256Hex(sessionID + ":" + promptCacheKey)
}

func newSyntheticStateKey(promptCacheKey string) string {
	return "anon-" + sha256Hex(promptCacheKey + ":" + randomHex(8) + ":" + strconv.FormatInt(time.Now().UnixNano(), 10))[:32]
}

func randomHex(size int) string {
	randomBytes := make([]byte, size)
	if _, err := rand.Read(randomBytes); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(randomBytes)
}

func isHashPrefix(previousHashes, nextHashes []string) bool {
	if len(previousHashes) > len(nextHashes) {
		return false
	}
	for index := range previousHashes {
		if previousHashes[index] != nextHashes[index] {
			return false
		}
	}
	return true
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	cloned := make([]string, len(values))
	copy(cloned, values)
	return cloned
}

func shortID(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "-"
	}
	if len(value) <= 20 {
		return value
	}
	return value[:10] + ".." + value[len(value)-8:]
}

func stableJSON(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case string:
		data, _ := json.Marshal(v)
		return string(data)
	case bool:
		if v {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case int:
		return strconv.Itoa(v)
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			parts = append(parts, stableJSON(item))
		}
		return "[" + strings.Join(parts, ",") + "]"
	case map[string]any:
		keys := make([]string, 0, len(v))
		for key := range v {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			parts = append(parts, stableJSON(key)+":"+stableJSON(v[key]))
		}
		return "{" + strings.Join(parts, ",") + "}"
	default:
		data, _ := json.Marshal(v)
		return string(data)
	}
}

func sha256Hex(input string) string {
	sum := sha256.Sum256([]byte(input))
	return hex.EncodeToString(sum[:])
}

func marshalString(value any) string {
	data, _ := json.Marshal(value)
	return string(data)
}
