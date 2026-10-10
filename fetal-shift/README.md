# fetal-shift

Does uncertainty-based deferral keep fetal ultrasound AI safe when it moves from Spanish hospitals
to African clinics, through tele-ultrasound compression and on-device quantization?

Samuel and Akachi. Target: TARPS 2027 abstract (Track 4, due 23 Oct 2026) and MIDL 2027.
The proposal and the ticket-by-ticket task list are shared separately; this README covers setup only.

## Data (both CC BY 4.0)

| Dataset | Role | Source |
| --- | --- | --- |
| FETAL_PLANES_DB (Spain) | Training, validation, in-distribution test | https://zenodo.org/records/3904280 |
| African fetal planes (5 countries) | External test only | https://zenodo.org/records/7540448 |

## Where things live

- **GitHub (this repo):** code, configs, frozen split CSVs. Nothing large.
- **Shared Drive folder `fetal-shift-drive/`:** data, checkpoints, model outputs, results.

```
fetal-shift-drive/
  data/zips/         # original Zenodo downloads (spain/, africa/)
  data/raw/          # extracted once, for browsing; read-only
  data/compressed/   # T3 output
  runs/              # T4 output (Akachi)
  outputs/           # T5 output (Akachi)
  eval/              # T7-T8 output (Samuel)
```

All paths inside CSVs are relative (e.g. `data/raw/africa/...`). `src/paths.py` resolves them
against the fast local copy on the Colab machine, falling back to Drive.

## First-time setup

**Samuel (once):**
1. Create the GitHub repo, push this folder, add Akachi as a collaborator.
2. In Google Drive, create `fetal-shift-drive` in *My Drive* and share it with Akachi as *Editor*.
3. Open `notebooks/colab_template.ipynb` in Colab, set `REPO_URL`, uncomment cell 5 and run all
   cells once. This creates the folders, downloads both datasets and extracts them.

**Akachi (once):**
1. Accept the GitHub invitation.
2. In Drive, *Shared with me* -> right-click `fetal-shift-drive` -> *Organize* -> *Add shortcut* -> *My Drive*.
3. Create a GitHub personal access token with read access to this repo and add it to Colab
   *Secrets* as `GITHUB_TOKEN`.
4. Run the Colab template; cell 7 should report about 12,400 Spanish and 450 African images.

## Every session

1. Open the Colab template and run cells 1-4, 6 and 7 (setup, pull latest code, local data copy).
2. Run your ticket's script.
3. Save anything that must persist to Drive, never to `/content` (wiped at session end).

## Rules

- Edit code in VS Code, commit and push; Colab only pulls and runs.
- Never commit data, checkpoints or outputs (see `.gitignore`).
- Never put the GitHub token in a notebook; use Colab Secrets.
- Labels: 0 Fetal abdomen, 1 Fetal brain, 2 Fetal femur, 3 Fetal thorax.
- Run IDs: `<model>_s<seed>`, e.g. `effb0_s3`. Seeds 0-4.
- Each person writes only to their own Drive folders.
