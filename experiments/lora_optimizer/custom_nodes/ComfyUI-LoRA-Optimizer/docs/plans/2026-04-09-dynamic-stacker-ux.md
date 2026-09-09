# Dynamic Stacker UX — Enable/Disable, Remove, Reorder

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-slot enable/disable toggle (grays out sibling widgets, saved with workflow) and right-click remove/move-up actions to `LoRAStackDynamic`.

**Architecture:** Python adds `enabled_{i}` BOOLEAN widgets and skips disabled slots in `build_stack`. JS overrides widget draw callbacks for gray-out, and hooks `node.getExtraMenuOptions` for remove/reorder using `widget.last_y` to identify the targeted slot.

**Tech Stack:** Python (lora_optimizer.py), LiteGraph/ComfyUI JS (js/lora_stack_dynamic.js)

---

### Key file locations

- **Python node:** `lora_optimizer.py` lines 366–531 (`LoRAStackDynamic` class)
  - `INPUT_TYPES()` widget definitions: lines 394–436
  - `build_stack()`: lines 494–531
- **JS extension:** `js/lora_stack_dynamic.js`
  - `toggleWidget()`: line 6
  - `updateVisibility()`: line 92
  - `interceptWidgetValue()`: line 33
  - `nodeCreated` registration: line 306
- **Tests:** `tests/test_lora_optimizer.py`

### Slot widget names (all 10 slots, replacing `{i}` with 1–10)

Per-slot widgets (in order they appear in the node):
- `enabled_{i}` ← **new**
- `lora_name_{i}` (dropdown)
- `lora_name_text_{i}` (text)
- `strength_{i}` (simple mode only)
- `model_strength_{i}` (advanced mode only)
- `clip_strength_{i}` (advanced mode only)
- `conflict_mode_{i}` (advanced mode only)
- `key_filter_{i}` (advanced mode only)

---

### Task 1: Python — add `enabled_{i}` widgets + skip in `build_stack`

**Files:**
- Modify: `lora_optimizer.py`
- Test: `tests/test_lora_optimizer.py`

**Step 1: Write the failing test**

Add to `LoRAOptimizerTests` in `tests/test_lora_optimizer.py`:

```python
def test_build_stack_skips_disabled_slots(self):
    from unittest import mock
    node = lora_optimizer.LoRAStackDynamic()
    # Slot 2 is disabled — must be excluded from output
    with mock.patch.object(
        lora_optimizer.LoRAStackDynamic, "_resolve_lora_name",
        side_effect=lambda n: n,  # return name as-is (bypass file lookup)
    ):
        result, = node.build_stack(
            settings_visibility="simple",
            input_mode="text",
            lora_count=3,
            lora_name_text_1="lora_a",
            lora_name_text_2="lora_b",
            lora_name_text_3="lora_c",
            strength_1=1.0,
            strength_2=0.8,
            strength_3=0.5,
            enabled_1=True,
            enabled_2=False,
            enabled_3=True,
        )
    names = [entry[0] for entry in result]
    self.assertEqual(len(result), 2)
    self.assertIn("lora_a", names)
    self.assertIn("lora_c", names)
    self.assertNotIn("lora_b", names)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_lora_optimizer.py::LoRAOptimizerTests::test_build_stack_skips_disabled_slots -v
```

Expected: FAIL — `build_stack` doesn't read `enabled_{i}` yet, so all 3 slots are returned.

**Step 3: Add `enabled_{i}` to `INPUT_TYPES()`**

In `lora_optimizer.py`, inside the `for i in range(1, cls.MAX_LORAS + 1):` loop (currently starting at line 394), add `enabled_{i}` as the **first entry** before `lora_name_{i}`:

```python
for i in range(1, cls.MAX_LORAS + 1):
    inputs["required"][f"enabled_{i}"] = ("BOOLEAN", {
        "default": True,
        "tooltip": f"Enable or disable LoRA #{i} without removing it from the list.",
    })
    inputs["required"][f"lora_name_{i}"] = (loras, {
    # ... rest unchanged
```

**Step 4: Skip disabled slots in `build_stack()`**

In `build_stack()`, add the enabled check as the first thing inside the loop body (before the name lookup). The loop currently starts: `for i in range(1, lora_count + 1):`. Change to:

```python
for i in range(1, lora_count + 1):
    if not kwargs.get(f"enabled_{i}", True):
        continue
    if use_text:
    # ... rest unchanged
```

**Step 5: Run the test**

```bash
python -m pytest tests/test_lora_optimizer.py::LoRAOptimizerTests::test_build_stack_skips_disabled_slots -v
```

Expected: PASS

**Step 6: Run full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

**Step 7: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add enabled_{i} toggle to LoRAStackDynamic — skips disabled slots in build_stack"
```

---

### Task 2: JS — gray-out disabled slots

**Files:**
- Modify: `js/lora_stack_dynamic.js`

No automated tests — verify manually by loading ComfyUI, toggling the enabled checkbox, and confirming sibling widgets visually gray out / restore.

**Step 1: Add `setSlotGrayed(node, i, grayed)` helper**

Add this function after the `patchLoraDisplayValue` block (around line 302), before the `// --- Node Registration ---` comment:

```js
const GRAY_ALPHA = 0.35;

function setSlotGrayed(node, i, grayed) {
    const siblings = [
        `lora_name_${i}`,
        `lora_name_text_${i}`,
        `strength_${i}`,
        `model_strength_${i}`,
        `clip_strength_${i}`,
        `conflict_mode_${i}`,
        `key_filter_${i}`,
    ];
    for (const name of siblings) {
        const w = findWidget(node, name);
        if (!w) continue;
        if (grayed) {
            if (!w._origDraw) {
                w._origDraw = w.draw?.bind(w) ?? null;
            }
            w.draw = function (ctx, node, width, y, height) {
                ctx.save();
                ctx.globalAlpha = GRAY_ALPHA;
                if (w._origDraw) w._origDraw(ctx, node, width, y, height);
                ctx.restore();
            };
        } else {
            if (w._origDraw !== undefined) {
                w.draw = w._origDraw ?? undefined;
                delete w._origDraw;
            }
        }
    }
}
```

**Step 2: Wire up `enabled_{i}` intercept in `nodeCreated`**

Inside the `nodeCreated` callback for `LoRAStackDynamic` (currently lines 308–331), add interception of `enabled_{i}` widgets. Add after the existing `for (const w of node.widgets || [])` loop:

```js
// Intercept enabled_{i} toggles for gray-out
for (let i = 1; i <= 10; i++) {
    const enabledW = findWidget(node, `enabled_${i}`);
    if (enabledW) {
        interceptWidgetValue(enabledW, (newVal) => {
            setSlotGrayed(node, i, !newVal);
            app.canvas?.setDirty?.(true, true);
        });
    }
}
```

**Step 3: Apply gray state on initial load in `updateVisibility`**

At the end of `updateVisibility()` (after the `filterWidget` block, before the resize lines), add:

```js
// Apply gray state for disabled slots
for (let i = 1; i <= MAX; i++) {
    const enabledW = findWidget(node, `enabled_${i}`);
    if (enabledW) {
        setSlotGrayed(node, i, !enabledW.value);
    }
}
```

**Step 4: Also hide `enabled_{i}` for slots beyond `lora_count`**

In `updateVisibility()`, inside the `for (let i = 1; i <= MAX; i++)` loop, add:

```js
toggleWidget(node, findWidget(node, `enabled_${i}`), visible);
```

as the first line inside that loop body (before the existing `toggleWidget` calls).

**Step 5: Manual verification**

Load ComfyUI → create a LoRAStackDynamic node → uncheck the `enabled_1` toggle → confirm all sibling widgets (lora name, strength, etc.) render at reduced opacity. Re-check → confirm they restore to full opacity. Save workflow → reload → confirm enabled state is preserved.

**Step 6: Commit**

```bash
git add js/lora_stack_dynamic.js
git commit -m "feat: gray out disabled LoRA slots in LoRAStackDynamic"
```

---

### Task 3: JS — right-click remove + move up

**Files:**
- Modify: `js/lora_stack_dynamic.js`

**Background:** LiteGraph calls `node.getExtraMenuOptions(canvas, options)` on right-click. `canvas.graph_mouse` is `[x, y]` in canvas coordinates. Each widget has `widget.last_y` set during the last draw pass, also in canvas coordinates. Comparing `widget.last_y` to `canvas.graph_mouse[1]` identifies the closest widget, and from that, the slot index.

**Step 1: Add `SLOT_WIDGET_NAMES` constant**

Add near the top of the file (after `const HIDDEN_TAG = ...`):

```js
const SLOT_WIDGET_NAMES = [
    "enabled",
    "lora_name",
    "lora_name_text",
    "strength",
    "model_strength",
    "clip_strength",
    "conflict_mode",
    "key_filter",
];
```

**Step 2: Add `getSlotAtY(node, mouseY)` helper**

Add after `setSlotGrayed`:

```js
function getSlotAtY(node, mouseY) {
    const countWidget = findWidget(node, "lora_count");
    const count = countWidget ? countWidget.value : 10;
    let bestSlot = null;
    let bestDist = Infinity;

    for (let i = 1; i <= count; i++) {
        for (const base of SLOT_WIDGET_NAMES) {
            const w = findWidget(node, `${base}_${i}`);
            if (!w || w.last_y === undefined || w.hidden) continue;
            const dist = Math.abs(w.last_y - mouseY);
            if (dist < bestDist) {
                bestDist = dist;
                bestSlot = i;
            }
        }
    }
    // Only return a slot if mouse is within 30px of a widget
    return bestDist < 30 ? bestSlot : null;
}
```

**Step 3: Add `copySlotValues(node, fromIdx, toIdx)` helper**

Copies all widget values from slot `fromIdx` to slot `toIdx`:

```js
function copySlotValues(node, fromIdx, toIdx) {
    for (const base of SLOT_WIDGET_NAMES) {
        const src = findWidget(node, `${base}_${fromIdx}`);
        const dst = findWidget(node, `${base}_${toIdx}`);
        if (src && dst) dst.value = src.value;
    }
}
```

**Step 4: Add `clearSlotValues(node, idx)` helper**

Resets a slot to defaults:

```js
function clearSlotValues(node, idx) {
    const defaults = {
        [`enabled_${idx}`]: true,
        [`lora_name_${idx}`]: "None",
        [`lora_name_text_${idx}`]: "None",
        [`strength_${idx}`]: 1.0,
        [`model_strength_${idx}`]: 1.0,
        [`clip_strength_${idx}`]: 1.0,
        [`conflict_mode_${idx}`]: "all",
        [`key_filter_${idx}`]: "all",
    };
    for (const [name, val] of Object.entries(defaults)) {
        const w = findWidget(node, name);
        if (w) w.value = val;
    }
}
```

**Step 5: Add right-click menu to `nodeCreated`**

Add after the `setTimeout(...)` block in the `LoRAStackDynamic` `nodeCreated` callback:

```js
const origGetExtra = node.getExtraMenuOptions;
node.getExtraMenuOptions = function (canvas, options) {
    if (origGetExtra) origGetExtra.apply(this, arguments);

    const countWidget = findWidget(this, "lora_count");
    const count = countWidget ? countWidget.value : 1;
    const mouseY = canvas.graph_mouse[1];
    const slot = getSlotAtY(this, mouseY);
    if (slot === null) return;

    const node_ = this;
    options.splice(-1, 0, null,
        {
            content: `Remove LoRA #${slot}`,
            callback() {
                // Shift slots [slot+1 .. count] up by one
                for (let i = slot; i < count; i++) {
                    copySlotValues(node_, i + 1, i);
                }
                clearSlotValues(node_, count);
                if (countWidget && count > 1) countWidget.value = count - 1;
                updateVisibility(node_);
                app.canvas?.setDirty?.(true, true);
            },
        },
        slot > 1
            ? {
                content: `Move LoRA #${slot} up`,
                callback() {
                    // Swap slot and slot-1
                    const tmp = {};
                    for (const base of SLOT_WIDGET_NAMES) {
                        const w = findWidget(node_, `${base}_${slot}`);
                        if (w) tmp[base] = w.value;
                    }
                    copySlotValues(node_, slot - 1, slot);
                    for (const base of SLOT_WIDGET_NAMES) {
                        const w = findWidget(node_, `${base}_${slot - 1}`);
                        if (w && tmp[base] !== undefined) w.value = tmp[base];
                    }
                    updateVisibility(node_);
                    app.canvas?.setDirty?.(true, true);
                },
            }
            : null,
    ).filter(Boolean);
};
```

Wait — `options.splice(-1, 0, ...)` mutates the array directly, and passing `null` for the "Move up" entry when slot is 1 will add a `null` separator with no following item. Use a conditional push instead:

```js
node.getExtraMenuOptions = function (canvas, options) {
    if (origGetExtra) origGetExtra.apply(this, arguments);

    const countWidget = findWidget(this, "lora_count");
    const count = countWidget ? countWidget.value : 1;
    const mouseY = canvas.graph_mouse[1];
    const slot = getSlotAtY(this, mouseY);
    if (slot === null) return;

    const node_ = this;
    const extras = [
        null,
        {
            content: `Remove LoRA #${slot}`,
            callback() {
                for (let i = slot; i < count; i++) {
                    copySlotValues(node_, i + 1, i);
                }
                clearSlotValues(node_, count);
                if (countWidget && count > 1) countWidget.value = count - 1;
                updateVisibility(node_);
                app.canvas?.setDirty?.(true, true);
            },
        },
    ];
    if (slot > 1) {
        extras.push({
            content: `Move LoRA #${slot} up`,
            callback() {
                const tmp = {};
                for (const base of SLOT_WIDGET_NAMES) {
                    const w = findWidget(node_, `${base}_${slot}`);
                    if (w) tmp[base] = w.value;
                }
                copySlotValues(node_, slot - 1, slot);
                for (const base of SLOT_WIDGET_NAMES) {
                    const w = findWidget(node_, `${base}_${slot - 1}`);
                    if (w && tmp[base] !== undefined) w.value = tmp[base];
                }
                updateVisibility(node_);
                app.canvas?.setDirty?.(true, true);
            },
        });
    }
    options.splice(-1, 0, ...extras);
};
```

**Step 6: Manual verification**

Load ComfyUI → create a LoRAStackDynamic node with 3+ LoRAs filled in → right-click on slot 2 → confirm "Remove LoRA #2" shifts slot 3 to slot 2 and decrements count → right-click on slot 2 again → confirm "Move LoRA #2 up" swaps it with slot 1. Verify that enabled state, name, and strength all move correctly.

Also verify: right-clicking the node header (not a slot) shows no slot-specific entries.

**Step 7: Commit**

```bash
git add js/lora_stack_dynamic.js
git commit -m "feat: add right-click remove and move-up to LoRAStackDynamic"
```

---

### Task 4: Final verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

**Step 2: End-to-end manual check**

1. Create LoRAStackDynamic with 4 slots, all filled.
2. Disable slot 2 → confirm it grays out.
3. Queue → confirm disabled slot is excluded from the stack output.
4. Right-click slot 3 → Remove → confirm shift up.
5. Right-click slot 2 → Move up → confirm swap with slot 1.
6. Save workflow, reload → confirm enabled states are preserved.

**Step 3: Run finishing-a-development-branch skill**
