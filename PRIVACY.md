# Publication boundary

This repository contains sanitized MiniCPM and LFM case studies plus a standalone Qwen single-GPU report. Each is built from an explicit allowlist of numeric results, fixed public labels and reviewed artifact hashes. Original private evidence is preserved separately.

Included material identifies the public hardware/model/runtime versions and the experiment configuration needed to interpret the result. It contains individual clean-run measurements, aggregate timing, test counts, equality flags and artifact hashes. The research document links public upstream source and AMD documentation.

Excluded material includes personal names and handles, local account names, hostnames, absolute machine paths, network addresses, authentication material, device serials, hardware instance IDs, raw prompts, answers, token IDs, generated edits, private runner configuration and complete logs. No original result object or log directory is copied wholesale.

The experiment's model and runtime hashes are included for provenance. Prompt, response, local launcher-configuration and device-identity hashes are not published. The private evidence manifest is not copied because its file inventory is outside the publication boundary.

## Check the public bundle

Run from this folder with an existing Python installation:

```text
python verify.py
python verify.py --deny-term PRIVATE_TERM
python verify.py --self-test
```

The verifier uses Python's standard library. It does not start inference, load GPU libraries, access the protected agent harness, contact a network service or publish anything. It checks the exact file inventory, numeric accounting, CSV/JSON agreement, relative links, content hashes and common path/address/credential/identity patterns. It reads the current account and host names only to reject their accidental inclusion; it does not save or print those values. Optional private terms strengthen the local check.

The verifier reports only a filename and problem category, not matched private text. A bounded scanner cannot prove that arbitrary prose contains no identifying information, so allowlisted extraction and a manual review of the complete bundle remain part of the process.

Private exporters write only explicitly selected data fields. They and their source-location receipts remain outside the publication folder. Public users can verify the supplied evidence; recreating it from private runs requires the retained original artifacts.

## Sharing scope

Only the twelve reviewed publication files are included in this revision. The surrounding experiment workspace and its Git history are not included. The repository began with a fresh publication history; updates retain a project-name author and committer and a GitHub no-reply email address. GitHub displays the owning public account; private email addresses and local Git identity settings are not used in commit attribution.

The bundle is documentation and evidence, not a packaged backend release. Public reproducer code and its licensing/notice review are separate future work.
