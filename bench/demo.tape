# demo.tape — a VHS script (https://github.com/charmbracelet/vhs) that records
# the emberllm demo GIF used at the top of the README. Regenerate with:
#
#     vhs bench/demo.tape
#
# Kept in the repo so the demo is reproducible, not a one-off screen capture.
Output bench/demo.gif

Set FontSize 18
Set Width 1100
Set Height 640
Set Padding 16
Set Theme "Catppuccin Mocha"

Type "# emberllm — a 110M model on a plain CPU, no GPU"
Enter
Sleep 800ms

Type "./ember generate models/stories110M-q8.ember -p 'Once upon a time' -n 120 -t 0.8"
Enter
Sleep 6s

Type "# ~250 tok/s of Q8_0 inference in ~1500 lines of C11"
Enter
Sleep 2s
