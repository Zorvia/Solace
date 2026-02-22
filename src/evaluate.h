--- /tmp/sf-orig/Stockfish-master/src/evaluate.h	2026-02-18 20:46:57.000000000 +0000
+++ /home/claude/Stockfish-master/src/evaluate.h	2026-02-22 07:25:19.672763463 +0000
@@ -36,6 +36,22 @@
 #define EvalFileDefaultNameBig "nn-5227780996d3.nnue"
 #define EvalFileDefaultNameSmall "nn-37f18f62d772.nnue"
 
+// ── Solace aggression control ────────────────────────────────────────────────
+// AggressionMode selects which aggression mechanism is active.
+//   OFF   — identical to baseline Stockfish evaluation (default).
+//   PARAM — runtime-tunable evaluation bias via set_aggression(0..100).
+//   NNUE  — reserved for a future aggression-trained NNUE net.
+enum class AggressionMode { OFF, PARAM, NNUE };
+
+// Set/get the engine-wide aggression level used when mode == PARAM.
+// level must be in [0, 100]; values outside that range are clamped.
+// Thread-safe (backed by std::atomic).
+void set_aggression(int level);
+int  get_aggression();
+void set_aggression_mode(AggressionMode mode);
+AggressionMode get_aggression_mode();
+// ── End Solace aggression control ────────────────────────────────────────────
+
 namespace NNUE {
 struct Networks;
 struct AccumulatorCaches;
