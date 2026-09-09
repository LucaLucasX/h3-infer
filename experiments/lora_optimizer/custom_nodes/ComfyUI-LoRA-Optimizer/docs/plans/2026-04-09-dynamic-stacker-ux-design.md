# Dynamic LoRA Stacker UX — Enable/Disable, Remove, Reorder

## Context

`LoRAStackDynamic` pre-defines 10 slots in Python `INPUT_TYPES()` and uses `lora_stack_dynamic.js` to toggle visibility based on `lora_count`. All slots always exist in the widget list; only their visibility changes.

## Requirements

1. **Enable/disable toggle per slot** — saves with workflow, grays out sibling widgets when off, does not decrement `lora_count`
2. **Right-click → Remove LoRA** — shifts all slots below up by one, decrements `lora_count`
3. **Right-click → Move up** — swaps all widget values with the slot above (no-op on slot 1)

## Design

### Python (`lora_optimizer.py`)

- Add `enabled_{i}` BOOLEAN widget (default `True`) as the **first widget of each slot**, before `lora_name_{i}`, in `INPUT_TYPES()` for all 10 slots.
- In `build_stack()`, skip any slot where `enabled_{i}` is `False`.
- No other Python changes required.

### JS: Gray-out on disable (`lora_stack_dynamic.js`)

- For each slot `i`, call `interceptWidgetValue` on `enabled_{i}`.
- On change to `false`: wrap each sibling widget's `draw` function — save original as `widget._origDraw`, replace with a wrapper that sets `ctx.globalAlpha = 0.4` before calling original, restores after.
- On change to `true`: restore `widget._origDraw` on each sibling.
- The `enabled_{i}` toggle widget itself always draws normally.
- Sibling widgets = all widgets in the slot except `enabled_{i}` itself: `lora_name_{i}`, `strength_{i}` / `model_strength_{i}` / `clip_strength_{i}`, `conflict_mode_{i}`, `key_filter_{i}`.

### JS: Right-click menu (`lora_stack_dynamic.js`)

- Override `node.onGetExtraMenuOptions(menuOptions, event)`.
- Compute target slot from mouse Y:
  - Get node-local Y = `event.canvasY - node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT`
  - Accumulate `LiteGraph.NODE_WIDGET_HEIGHT` per visible widget until accumulated height exceeds node-local Y → that widget's slot index is the target
- Add menu entries only when a valid active slot is targeted:
  - **"Remove LoRA"**: copy widget values from slots `[target+1 .. lora_count-1]` to `[target .. lora_count-2]` (all fields: `enabled`, name, strengths, conflict_mode, key_filter), clear the last active slot's values, decrement `lora_count`, call `updateVisibility(node)`.
  - **"Move up"**: if target > 0, swap all widget values between slot `target` and slot `target-1`.
- Widget values to copy/swap per slot: `enabled_{i}`, `lora_name_{i}`, `strength_{i}`, `model_strength_{i}`, `clip_strength_{i}`, `conflict_mode_{i}`, `key_filter_{i}`.

## Constraints

- Slot shifting operates on widget `.value` directly — no server round-trip needed.
- `updateVisibility()` already handles re-hiding/showing widgets after `lora_count` changes.
- Both dropdown and text input modes must be handled when copying name values.
