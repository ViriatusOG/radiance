# Compose files

Three fleets, each one container serving several models behind one port and bringing a model onto
its cards when a request names it ([`docs/MULTIMODEL.md`](../../docs/MULTIMODEL.md)). Copy this
directory to the serving host, fill in `.env` from `.env.example`, and run one of them
([`docs/DOCKER.md`](../../docs/DOCKER.md) has the details):

```sh
docker compose -f tp2.yaml up -d      # start; it comes back after a reboot
docker compose -f tp2.yaml logs -f    # follow the log
docker compose -f tp2.yaml down       # stop
```

| compose file | cards | models |
|---|---|---|
| `tp1.yaml` | one (`RADIANCE_GPU`) | Qwen3.8-Flash-Next, Qwen3.8-27B IQ4R-L, MiniCPM5-2B, EmbeddingGemma 2, Nemotron 3.5 Rerank 8B, Qwen3-ASR 1.7B, Breeze-TTS-2, d1-3B, Qwen-Image 2.1, MiniMax-Music3, Hunyuan3D-2 |
| `tp2.yaml` | two | Qwen3.8-Flash-Next, Qwen3.8-27B IQ8R, MiniCPM5-2B, EmbeddingGemma 2, Nemotron 3.5 Rerank 8B, Qwen3-ASR 1.7B, Breeze-TTS-2, Qwen-Image 2.1, MiniMax-H3, MiniMax-Music3, Hunyuan3D-2 |
| `tp4.yaml` | four | GLM-5.3-Flash (`server.kernels` puts `glm` first) |

Each compose file has its server's configuration beside it, `tp1.radiance.yaml`,
`tp2.radiance.yaml` and `tp4.radiance.yaml`: plain radiance YAML (`radiance --config tp2.radiance.yaml --check` validates
it, with `RADIANCE_API_KEY` set), which Compose places in the container. An entry's name under `models:` is the model id a
request gives, and its `model:` is a file or a checkpoint directory under `RADIANCE_MODELS`, as the
top-level README's model table downloads or converts it. Delete the entries of the models you do
not have, and add any other model the README lists with its keys from there. The configuration's
header says what the fleet needs of the host and what a model costs to bring back; its
`server.port` (8000) is the port, which the compose file's health check asks as well.

`seccomp-radiance.json` is the profile `RADIANCE_SECCOMP` may name; `gemma3.jinja` is the chat
template TranslateGemma is served with (`override-chat-template`).
