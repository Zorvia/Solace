--- /tmp/sf-orig/Stockfish-master/src/evaluate.cpp	2026-02-18 20:46:57.000000000 +0000
+++ /home/claude/Stockfish-master/src/evaluate.cpp	2026-02-22 07:26:36.779762623 +0000
@@ -19,6 +19,7 @@
 #include "evaluate.h"
 
 #include <algorithm>
+#include <atomic>
 #include <cassert>
 #include <cmath>
 #include <cstdlib>
@@ -37,6 +38,27 @@
 
 namespace Stockfish {
 
+// ── Solace aggression state ──────────────────────────────────────────────────
+// All reads/writes are lock-free atomics; no search-thread coordination needed.
+namespace {
+std::atomic<int>                  g_aggression_level{0};
+std::atomic<Eval::AggressionMode> g_aggression_mode{Eval::AggressionMode::OFF};
+}
+
+void Eval::set_aggression(int level) {
+    g_aggression_level.store(std::clamp(level, 0, 100), std::memory_order_relaxed);
+}
+int Eval::get_aggression() {
+    return g_aggression_level.load(std::memory_order_relaxed);
+}
+void Eval::set_aggression_mode(Eval::AggressionMode mode) {
+    g_aggression_mode.store(mode, std::memory_order_relaxed);
+}
+Eval::AggressionMode Eval::get_aggression_mode() {
+    return g_aggression_mode.load(std::memory_order_relaxed);
+}
+// ── End Solace aggression state ──────────────────────────────────────────────
+
 // Returns a static, purely materialistic evaluation of the position from
 // the point of view of the side to move. It can be divided by PawnValue to get
 // an approximation of the material advantage on the board in terms of pawns.
@@ -78,10 +100,33 @@
     nnue -= nnue * nnueComplexity / 18236;
 
     int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
-    int v        = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;
 
-    // Damp down the evaluation linearly when shuffling
-    v -= v * pos.rule50_count() / 199;
+    // Solace PARAM mode: boost optimism weight to increase preference for
+    // dynamic, imbalanced positions. At aggression=0 this is a no-op.
+    int optimism_material_bias = 7191;
+    if (g_aggression_mode.load(std::memory_order_relaxed) == AggressionMode::PARAM)
+    {
+        int agg = g_aggression_level.load(std::memory_order_relaxed);
+        // Scale bias from 7191 (baseline) up toward 14382 (+100%) at agg=100.
+        optimism_material_bias += agg * 7191 / 100;
+    }
+
+    int v = (nnue * (77871 + material) + optimism * (optimism_material_bias + material)) / 77871;
+
+    // Damp down the evaluation linearly when shuffling.
+    // Solace PARAM mode: aggression level reduces this damping, making the
+    // engine less draw-averse and more willing to press dynamic positions.
+    // At aggression=0 the arithmetic is identical to baseline.
+    {
+        int rule50_divisor = 199;
+        if (g_aggression_mode.load(std::memory_order_relaxed) == AggressionMode::PARAM)
+        {
+            int agg = g_aggression_level.load(std::memory_order_relaxed);
+            // Scale divisor from 199 (baseline) up to 398 (half damping) at agg=100.
+            rule50_divisor += agg * 199 / 100;
+        }
+        v -= v * pos.rule50_count() / rule50_divisor;
+    }
 
     // Guarantee evaluation does not hit the tablebase range
     v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
