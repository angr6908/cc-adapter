# cc-adapter

GPT (OpenAI API) adapter for Claude Code, featuring a standalone Go binary experience.

## Support
- Upstream Model (e.g. gpt-5.4)
- Reasoning Effort (e.g. xhigh)
- Fast Mode (true/false)
- Context Window (e.g. 1000000)
- Advanced Prompt Caching (enabled by default)
- High-Efficiency Streaming and SSE (enabled by default)
- Session Retention (enabled by default)

## Config

The binary looks for config files in this order:

1. `config.local.json` next to the binary
2. `config.json` next to the binary
3. `config.local.json` in the current working directory
4. `config.json` in the current working directory

The gateway persists compact runtime state to `session-state.json` next to the config file.
That file stores the generated `cache_id` plus recent compact session metadata used by the gateway.
You can override the path with `CLAUDE_PROXY_SESSION_STATE_PATH`.

## Run

1. Copy `config.sample.json` to `config.local.json`
2. Fill in your tokens and endpoint
3. Start the gateway:

```sh
chmod +x cc-adapter && ./cc-adapter
```

## Claude Code setup

The gateway writes a convenience `claude-settings.json` into the same folder at startup. You can use it as a reference for manual setup.
