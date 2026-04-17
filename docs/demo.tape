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

Type "# A real differentiation run, replayed from its evolution journal."
Enter
Type ".venv/bin/stem-agent journal --path docs/example_run/journal.json | head -36"
Enter
Sleep 5s

Type "clear" Enter
Type "# Baseline F1 0.000 -> Specialized F1 0.778 on the 20-sample benchmark."
Enter
Type "grep -E 'baseline_f1|specialized_f1|f1_delta' docs/example_run/journal.json | head -8"
Enter
Sleep 4s
