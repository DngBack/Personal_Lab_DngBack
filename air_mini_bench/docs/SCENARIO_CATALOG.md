# Scenario catalog — tra cứu nhanh

Bảng đầy đủ các suite.

- **README tổng hợp:** [README.md](../README.md)
- **Giải thích sâu:** [HUONG_DAN_BENCH.md](./HUONG_DAN_BENCH.md)

---

## Phase 2 — Priority (6)

| Suite | Conv | Tool | LC | Arrival | Cache | Output bias |
|-------|------|------|-----|---------|-------|-------------|
| `official_like` | 31% | 59% | 10% | `official_window` (mean 50ms) | — | default |
| `steady_poisson` | 31% | 59% | 10% | `steady_poisson` (50ms) | — | default |
| `microburst` | 31% | 59% | 10% | `microburst` (6–14 / wave) | — | default |
| `tool_cache_hot` | 10% | 85% | 5% | session + **same ts** | **hot** | default |
| `decode_pressure` | 80% | 20% | 0% | `steady_poisson` (~35ms) | — | **long_decode** |
| `long_context_pressure` | 10% | 40% | 50% | `lc_spread_poisson` | — | default |

**Lệnh generate một suite:**

```bash
cd air_mini_bench && PYTHONPATH=src python -m bench.generate --phase phase2 --suite <TEN> --seed 42
```

**Lệnh chạy:**

```bash
PYTHONPATH=src python -m bench.run_bench --phase phase2 --suite <TEN>
```

---

## Phase 2 — Extended (4)

| Suite | Conv | Tool | LC | Arrival | Cache |
|-------|------|------|-----|---------|-------|
| `tool_cache_cold` | 10% | 85% | 5% | `session_cluster` | **cold** |
| `fast_queue` | 31% | 59% | 10% | `steady_poisson` (16ms) | — |
| `large_burst` | 20% | 70% | 10% | `large_burst` (10–25) | — |
| `flood_admission` | 40% | 55% | 5% | `flood` (t=0) | — |

```bash
PYTHONPATH=src python -m bench.generate --phase phase2 --all-suites --seed 42
```

---

## Phase 1 — All (6)

| Suite | Conv | Tool | Arrival | Cache | Output bias |
|-------|------|------|---------|-------|-------------|
| `p1_official_like` | 60% | 40% | `official_window` | — | default |
| `p1_steady` | 60% | 40% | `steady_poisson` (50ms) | — | default |
| `p1_burst` | 50% | 50% | `large_burst` (10–25) | — | default |
| `p1_tool_cache_hot` | 20% | 80% | session + same ts | **hot** | default |
| `p1_tool_cache_cold` | 20% | 80% | `session_cluster` | **cold** | default |
| `p1_decode_pressure` | 85% | 15% | `steady_poisson` (~30ms) | — | **long_decode** |

```bash
PYTHONPATH=src python -m bench.run_bench --phase phase1 --suite p1_steady
```

---

## Arrival kinds (implementation)

| `arrival` trong code | Hành vi timestamp |
|---------------------|-------------------|
| `steady_poisson` | Exponential gaps, tham số `mean_ms` |
| `official_window` | Lấy đoạn liên tục từ Poisson dài hơn |
| `microburst` | Sóng `wave_min`–`wave_max`, cùng `t` |
| `large_burst` | Sóng `batch_min`–`batch_max`, cùng `t` |
| `session_cluster` | Tool session: gap nhỏ trong session |
| `lc_spread_poisson` | Poisson + không trùng timestamp cho LC |
| `flood` | Tất cả `timestamp = 0` |

Định nghĩa đầy đủ: `src/bench/domain/scenario.py`, logic: `src/bench/arrivals.py`.
