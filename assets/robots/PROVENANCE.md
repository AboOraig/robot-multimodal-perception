# Robot asset provenance

* Model: Franka Emika Panda (arm + two-finger hand), `franka_emika_panda/panda.xml`
* Source: https://github.com/google-deepmind/mujoco_menagerie
* Pinned commit: `370c43965323c0944b433b202aac5058b5f907a9`
* Licence: Apache License 2.0 (see `franka_emika_panda/LICENSE` after fetching)
* Fetched by `scripts/fetch_panda.ps1`; the folder is git-ignored (~36 MB) and never modified.
  The table, cube, camera, TCP site and wrist force/torque sensor are added programmatically
  in `src/mmperc/simulation/scene.py` with `mujoco.MjSpec`.
