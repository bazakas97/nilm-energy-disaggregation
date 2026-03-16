<<<<<<< HEAD
# EnergyDiss



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://iti-gitlab.iti.gr/REFLEEX/energydiss.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://iti-gitlab.iti.gr/REFLEEX/energydiss/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
* [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
=======
# REEFLEX-NILM

Inference and training code for the REEFLEX NILM pipeline.

This repository is now organized around a small number of official entrypoints:

- `configs/active/release_eval.yaml`: main inference / evaluation config
- `configs/active/train_mains5_all10.yaml`: main training config
- `scripts/fetch_sel_daily.py`: fetch one day from SEL API
- `scripts/run_daily_eval.py`: generate per-house daily configs and run inference

Old experimental configs are still available under `configs/archive/`, but they are not the recommended starting point.

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you use CUDA, install the matching PyTorch build instead of the CPU wheel.

## Run inference / evaluation

Main command:

```bash
python run.py --config configs/active/release_eval.yaml
```

Before running, check these paths inside `configs/active/release_eval.yaml`:

- `paths.train_data`
- `paths.val_data`
- `paths.test_data`

The model bundle used for inference is already tracked in:

- `models/nilmformer_paper_mains5_all10_60_20_20/`

Outputs are written under `results/`:

- predictions CSV
- metrics JSON
- Plotly HTML plots

## Run training

Main training command:

```bash
python run.py --config configs/active/train_mains5_all10.yaml
```

This trains the current “official” NILMFormer-style configuration and saves outputs under `results/models/` and `results/plots/`.

## Daily SEL API inference

One-command daily pipeline:

```bash
python scripts/run_daily_pipeline.py \
  --date 2026-03-15 \
  --participants certhr5fwl7p,certhckoz1h4
```

This wrapper does:

1. fetch one day from SEL API
2. build the merged daily CSV
3. run per-house inference with `configs/active/release_eval.yaml`
4. automatically restrict reported outputs to the devices detected for each house from the fetched SEL sensors

Set credentials:

```bash
export SEL_API_EMAIL="you@example.com"
export SEL_API_PASSWORD="your-password"
```

Fetch one day:

```bash
python scripts/fetch_sel_daily.py \
  --date 2026-03-15 \
  --participants certhr5fwl7p,certhckoz1h4 \
  --output-dir DATA/daily_sel_api
```

Run per-house inference on the fetched merged CSV:

```bash
python scripts/run_daily_eval.py \
  --base-config configs/active/release_eval.yaml \
  --date 2026-03-15 \
  --split-data-csv DATA/daily_sel_api/20260315/daily_20260315_merged.csv \
  --per-house \
  --house-overrides configs/active/house_overrides_daily.example.yaml \
  --run
```

This daily inference path is sensor-aware:

- it reads the fetched `*_sensors.json` for each participant
- keeps only the devices that actually belong to that house
- writes plots / prediction columns / reported metrics only for that subset

Daily outputs are written to:

- `DATA/daily_sel_api/YYYYMMDD/`: fetched and normalized daily data
- `results/generated_configs/daily_eval_<split>_YYYYMMDD/`: generated per-house configs
- `results/csv/`: predictions CSV and metrics JSON
- `results/plots_*/YYYYMMDD/<participant>/`: Plotly HTML plots per house

If the daily CSV has no appliance labels, metrics are not meaningful. In that case, inspect the predictions CSVs and the HTML plots.

## What preprocessing is currently implemented

Fetch-time preprocessing in `scripts/fetch_sel_daily.py`:

- converts SEL energy values to power
- reindexes each participant/day to a fixed 1-minute grid
- interpolates only short gaps
- drops house/day when mains missing ratio is too high

Inference-time preprocessing in the active configs:

- `participant_data_filter`
- EV `label_gap_fill`
- `unattributed_mains_mask`
- participant/device gating
- postprocessing thresholds and denoising from the active config / house overrides

This means the pipeline is operational, but house-specific postprocessing is still configurable and not “final for every house forever”.

## Repository layout

- `configs/active/`: supported configs
- `configs/archive/`: legacy / experimental configs kept for reference
- `models/`: tracked inference model bundle
- `scripts/`: SEL fetch + daily run helpers
- `DATA/`: local datasets only, not tracked
- `results/`: generated outputs only, not tracked

## Docker

Docker support is available, but it is optional. See `DOCKER.md` if you want a containerized run.
>>>>>>> origin/repo-cleanup
