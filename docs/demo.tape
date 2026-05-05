Output docs/demo.mp4

Set Shell "zsh"
Set FontSize 14
Set Width 1100
Set Height 720
Set Theme "Catppuccin Mocha"
Set TypingSpeed 20ms
Set PlaybackSpeed 1.0

Hide
Type "setopt interactivecomments" Enter
Type "clear" Enter
Show

Type "# Replay the best cq seed of a 3-seed live run (11-sample validation slice)."
Enter
Type ".venv/bin/stem-agent journal --path docs/example_run/seeds/cq/journal_seed1.json | head -36"
Enter
Sleep 6s

Type "clear" Enter
Type "# Bootstrap mean and 95% CI pooled across the 3 cq seeds."
Enter
Type ".venv/bin/stem-agent report docs/example_run/seeds/cq"
Enter
Sleep 4s
