import { app } from "/scripts/app.js";

// LoRAMergeSettings: the auto_strength_floor's old -1 "auto" sentinel was split
// into a separate `auto_strength_floor_mode` switch, appended at the END of the
// node's widgets. litegraph restores widget values POSITIONALLY, so an old
// workflow (saved before the switch existed) has one fewer value — the new
// switch would default to "auto" and silently drop any explicit floor the user
// had set. Append the 8th value here, BEFORE positional restore, derived from
// the saved floor: an explicit floor (>= 0) -> "manual" (preserve it); the -1
// auto default -> "auto". New workflows already carry the value and are skipped.
const SETTINGS_OLD_COUNT = 7; // widgets before the mode switch was added
const FLOOR_INDEX = 2;        // position of auto_strength_floor in the widget order

app.registerExtension({
    name: "LoRAOptimizer.LoRAMergeSettings",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAMergeSettings") return;
        const origConfigure = node.configure;
        node.configure = function (info) {
            const wv = info && info.widgets_values;
            if (Array.isArray(wv) && wv.length === SETTINGS_OLD_COUNT) {
                const floor = wv[FLOOR_INDEX];
                const mode = typeof floor === "number" && floor >= 0 ? "manual" : "auto";
                wv.push(mode);
                console.log(
                    `[LoRAMergeSettings] Migrated workflow: auto_strength_floor_mode='${mode}' ` +
                    `(preserved prior floor=${floor}).`
                );
            }
            return origConfigure ? origConfigure.apply(this, arguments) : undefined;
        };
    },
});
