# NeutronStar (ds4) adoption plan

Plan to run GLM-5.2 / Tencent Hy3 / DeepSeek V4 on the local APUs via the
NeutronStar (ds4) ROCm engine, benchmark, and then resume the HOT-RAM KV work.

## Repo
Fork at `~/git/neutronstar` (upstream `antirez/ds4`, "DwarfStar"). Prep branch
`hx370-port` holds build/bench scripts + docs (install recipe, colibri
integration, KV-port design). The engine streams routed MoE experts from SSD
per token; `ds4-server` is the OpenAI-compatible frontend (`:8000`).

## Machines
- This box: HX 370 / Radeon 890M (gfx1150), 96 GB, no ROCm yet.
- Other: Strix Halo (gfx1151), 128 GB — the original `make strix-halo` target.

## Status
- [x] Pre-flight docs + build (`build-hx370.sh`) + bench (`bench-strix-halo.sh`)
      committed on `hx370-port`.
- [x] Models downloading to `/media/scott/NAS5/fileserver/model-binaries/`
      (DeepSeek-V4-Flash q2, Hy3 IQ2XXS, GLM-5.2 IQ2XXS). Resume-safe.
- [ ] ROCm 7.0 + new kernel install (held; user will install to mirror other
      machine).
- [ ] `build-hx370.sh` (needs ROCm + rocWMMA internal headers + 96 GB GTT
      kernel params).
- [ ] `bench-strix-halo.sh` speed numbers (HX 370 vs Strix Halo).
- [ ] Repoint colibri `[primary]/[helper]/[tiny.*]/[harness]` at ds4-server
      (config-only; see neutronstar `docs/COLIBRI-INTEGRATION.md`).
- [ ] Resume HOT-RAM KV port into `ds4_kvstore.c` (neutronstar `docs/KV-PORT.md`).

## colibri fleet (already done, commits on `hx370-benchmark`)
- 3-tier fleet (glm + hermes helper + tier-3 harness), `glm` CLI, proxy,
  `glm doctor`, all DISARMED behind `COLI_LIVE`.
- HOT-RAM + SSD KV tiers in `c/glm.c` (verified lossless prefill-skip).
- `tiny-harness train` drives `auto-finetune` for subagent models.

## Next
1. Let NAS downloads finish (DeepSeek is the slow one).
2. Follow the full procedural runbook in the neutronstar repo:
   `~/git/neutronstar/docs/RUNBOOK.md` — staging verify → ROCm install (reboot)
   → build → run ds4-server → repoint this fleet → benchmark → KV HOT-RAM port.
   Every phase has a verify step and explicit [HUMAN] checkpoints.
3. Build + bench; compare HX 370 vs Strix Halo.
4. Wire ds4-server into colibri config (see `configs/colibri-ds4.toml.example`);
   then KV port.
