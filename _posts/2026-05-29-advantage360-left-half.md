## Fixing an Intermittent Left-Half Failure on the Kinesis Advantage360

A follow-up to my [earlier post on programming a Dvorak layout for the Advantage360]({% post_url 2026-01-29-advantage360 %}). Same keyboard, different problem: the left half of my **wired** Advantage360 (KB360, SmartSet engine — not the Pro) started dropping out a few times a day. If you're seeing the same thing, this post is the short version of how I diagnosed it and what fixed it.

If you want the full debug log, kernel captures, hypotheses, and flashing notes, everything is in [github.com/mfaulk/kinesis-advantage360](https://github.com/mfaulk/kinesis-advantage360).

## The symptom

- Several times a day, the **entire left half** went unresponsive at once — not a column or row, the whole half.
- The right half kept working normally.
- No pattern I could tie it to: not sleep/resume, not typing intensity, not time of day, not bumping the keyboard.
- Recovery was always the same: unplug the host USB cable, plug it back in, left half returns.

The "all at once" failure mode ruled out progressive matrix wear, and the host USB connection was direct (no hub).

## Diagnosis

The first useful step was a 24-hour kernel-log capture:

```
journalctl --since "24 hours ago" | grep -iE 'usb|hid|0360|29ea'
```

This showed the Adv360 fully re-enumerating four times in 24 hours — matching the "several times a day" subjective experience. But the important finding was a **negative** one:

> There were **no USB errors preceding any of the re-enumerations**. No descriptor-read errors, no port resets, no link issues. The host saw a healthy device the entire time, then a clean disconnect/reconnect when I replugged.

That ruled out the host-side USB stack. Whatever was failing, it was happening *inside* the keyboard — the left half was silently dropping off the keyboard's internal bus, and only a full power-cycle brought it back.

Plausible suspects, in rough order:

1. **Inter-half USB-C cable or port** — intermittent contact on the cable that connects the two halves.
2. **Left-half MCU firmware lockup** — the left-half microcontroller hanging on stock firmware until power-cycled.

Swapping / jiggling the USB cable had no effect.

## The fix

I was on stock firmware (**1.0.69**), and the current release was **1.0.79** (10 minor versions newer). Updating to firmware **1.0.79** (via the SmartSet + v-Drive flow documented by Kinesis) appears to have resolved the issue. After roughly three weeks of normal use, the left half has not dropped out.

A few notes:

- **Check your version first.** With the cursor in a text editor, press **SmartSet + Right Shift + Right Ctrl + ?** for a Status Report. It will type out the L/R firmware versions. SmartSet shortcuts use the **stock** (default-layer) key positions, so if you've remapped any of those keys, press them at the physical position they occupy in the stock layout.
- **Back up your v-Drive before flashing.** The flash is supposed to preserve remaps, and in my case all 30 of mine survived intact, but a backup of `layouts/`, `macros/`, etc. is a few seconds of insurance against having to redo a custom layout.
- **The flash takes ~45 seconds. Don't unplug or type during it.** Cutting power mid-flash can brick a half.
- **One quirk worth flagging:** on 1.0.69, closing the v-Drive (SmartSet + v-Drive) after copying the `.upd` file hung the keyboard — right-half LEDs solid green, left-half LEDs dark, both halves unresponsive. Reproduced twice. The recovery was a USB replug; after that, you can skip reopening the v-Drive and go straight to **SmartSet + Right Ctrl + U** to install. The `.upd` file is already on the v-Drive's flash storage at that point.