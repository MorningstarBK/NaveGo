# pynavego

Bộ công cụ Python để **mô phỏng cảm biến quán tính (IMU)** và **định vị quán tính kết hợp thông tin độ cao** từ phong vũ kế (barometer), có thêm GNSS tùy chọn. Dự án học theo cấu trúc và mô hình toán của toolbox MATLAB NaveGo ở thư mục cha (Groves 2013; Titterton & Weston 2004).

## Cài đặt và chạy

```bash
cd python
pip install -e ".[plot,test]"
python examples/example_ins_baro.py          # bảng RMSE + ins_baro_results.png
python examples/example_ins_baro.py --imu consumer
python examples/example_allan.py
pytest -q
```

## Thành phần

| Module | Nội dung | Tương ứng NaveGo |
|---|---|---|
| `earth.py` | Mô hình WGS-84: bán kính cong, trọng lực, tốc độ quay Trái Đất, transport rate | `ins/` |
| `attitude.py` | Euler ↔ DCM, Rodrigues, log ma trận quay | `conversions/` |
| `ins.py` | Cơ học strapdown trong hệ NED | `ins/*_update.m` |
| `sim/trajectory.py` | Bộ sinh quỹ đạo tham chiếu (đứng yên → chạy, rẽ, lên/xuống dốc) | *(NaveGo không có)* |
| `sim/imu.py` | IMU lý tưởng từ quỹ đạo, thêm lỗi: bias tĩnh, bias động Gauss-Markov, ARW/VRW; preset `tactical`/`industrial`/`consumer` | `simulation/`, `imu_si_errors.m` |
| `sim/baro.py` | Phong vũ kế: độ cao thật + bias + trôi Gauss-Markov + nhiễu trắng | *(mới; nằm trong roadmap của NaveGo)* |
| `sim/mag.py` | Từ kế 3 trục: từ trường Trái Đất (F, độ từ khuynh, độ từ thiên) + bias hard-iron + nhiễu | `ins_gnss_mag.m` |
| `sim/gnss.py` | GNSS vị trí/vận tốc, có khoảng mất tín hiệu | `gnss_gen.m`, `gnss_outage.m` |
| `fusion.py` | EKF trạng thái lỗi 19 trạng thái, vòng kín (baro, từ kế, GNSS) | `ins-gnss/ins_gnss.m` |
| `metrics.py`, `allan.py` | Sai số/RMSE, độ lệch Allan | `performance-analysis/`, `allan-variance/` |

## Mô hình lọc

Vector trạng thái lỗi gồm: lỗi tư thế (3), lỗi vận tốc NED (3), lỗi vị trí NED tính bằng mét (3), bias con quay (3), bias gia tốc kế (3) và **bias phong vũ kế (1)**. Bộ lọc lan truyền ở tần số IMU và cập nhật khi có mẫu baro hoặc GNSS. Sau mỗi lần cập nhật, các hiệu chỉnh được đưa ngược vào INS rồi trạng thái lỗi được đặt lại về 0.

- Kênh đứng của INS thuần túy không ổn định (δv̇_D = 2g/R·δp_D), nên sai số độ cao tăng theo hàm mũ.
- Phép đo baro `z = h_INS − (h_baro − b̂) = −δp_D − δb` giữ cho sai số độ cao bị chặn.
- Khi chỉ có baro, bias của baro không quan sát được: độ cao bám theo baro nhưng vẫn lệch đúng bằng bias đó. Khi có GNSS, bộ lọc ước lượng được bias này, nên trong lúc GNSS mất tín hiệu baro vẫn giữ độ cao chính xác.

Kết quả mẫu với IMU `industrial`, 600 s, GNSS mất tín hiệu từ 300 đến 420 s:

```
scenario           h RMSE  h RMSE out   hor RMSE  hor RMSE out
INS               3992.85     3237.11   12332.95       8784.11
INS+Baro             1.63        2.10    1451.98       1574.64
INS+GNSS             1.96        3.88      11.23         24.81
INS+GNSS+Baro        0.74        0.60      10.41         22.91
```

Đơn vị là SI, góc tính bằng radian, hệ dẫn đường NED, và `C_nb` là ma trận quay từ hệ thân sang hệ dẫn đường. Chưa mô hình hóa cánh tay đòn (lever arm) của anten GNSS.

Giấy phép: LGPL v3, giống NaveGo.

## IMU quay quanh trục X / Y / Z + encoder (rotation modulation)

`sim/rotation.py` mô phỏng IMU gắn trên bàn xoay, với trục quay trùng trục X, Y hoặc Z của xe. Có hai chế độ quay: quay liên tục, hoặc quay đảo chiều ±360° (`--mode alternate`). Encoder đo góc quay θ (lượng tử hóa 16 bit, có nhiễu). INS/EKF ước lượng tư thế của khung cảm biến `C_ns`, sau đó góc roll/pitch/yaw của xe được tính bằng `C_nb = C_ns · R_trục(θ_enc)ᵀ`.

```bash
python examples/example_rotating_imu.py                 # quay quanh X, Y, Z + IMU cố định; GNSS mất từ 60 s
python examples/example_rotating_imu.py --axes z --imu consumer --rate 60 --mode alternate
```

Kết quả mẫu (IMU `industrial`, 600 s, chỉ có GNSS trong 60 s đầu; các thống kê tính trong lúc mất GNSS):

| IMU | Hỗ trợ | Trôi ngang cuối (m) | Trôi ngang max (m) | RMSE ngang (m) | Tốc độ trôi ngang (m/phút) | Sai số độ cao cuối (m) | RMSE độ cao (m) | RMSE Roll/Pitch/Yaw (°) |
|---|---|---|---|---|---|---|---|---|
| Quay X | Chỉ IMU | 7082.6 | 7082.6 | 2815.0 | 787.0 | 80.67 | 27.21 | 0.409 / 0.178 / 0.219 |
| Quay X | IMU + Baro | 417.4 | 684.7 | 303.4 | 46.4 | 0.23 | 0.56 | 0.130 / 0.094 / 0.242 |
| Quay Y | Chỉ IMU | 9641.4 | 9641.4 | 3862.9 | 1071.3 | -201.18 | 73.43 | 0.278 / 0.528 / 0.262 |
| Quay Y | IMU + Baro | 737.0 | 820.7 | 346.7 | 81.9 | 0.15 | 0.55 | 0.155 / 0.143 / 0.352 |
| Quay Z | Chỉ IMU | 2371.8 | 2371.8 | 1039.4 | 263.5 | 124.27 | 64.20 | 0.074 / 0.075 / 0.712 |
| Quay Z | IMU + Baro | 1614.8 | 1615.7 | 747.8 | 179.4 | 0.45 | 0.63 | 0.052 / 0.070 / 0.686 |
| Cố định | Chỉ IMU | 14213.3 | 14213.3 | 5649.6 | 1579.3 | 567.27 | 257.05 | 0.845 / 0.541 / 0.934 |
| Cố định | IMU + Baro | 2872.4 | 2876.8 | 1314.2 | 319.2 | 0.47 | 0.64 | 0.272 / 0.237 / 0.957 |

### Thay baro bằng từ kế (mặc định của `example_rotating_imu.py`)

Từ kế gắn cùng IMU nên quay theo IMU. Bộ lọc dùng cả vector từ trường: `z = C_nsᵀ m_n + b̂_m − m_đo = −Cᵀ[m_n×]φ − δb_m`, kèm 3 trạng thái bias hard-iron. Nhờ vậy sai số hướng (yaw) bị chặn, nhưng độ cao thì không được hỗ trợ. Chọn nguồn hỗ trợ bằng `--aid mag|baro|both`.

| IMU | Hỗ trợ | Trôi ngang cuối (m) | Trôi ngang max (m) | RMSE ngang (m) | Tốc độ trôi ngang (m/phút) | Sai số độ cao cuối (m) | RMSE độ cao (m) | RMSE Roll/Pitch/Yaw (°) |
|---|---|---|---|---|---|---|---|---|
| Quay X | Chỉ IMU | 7082.6 | 7082.6 | 2815.0 | 787.0 | 80.67 | 27.21 | 0.409 / 0.178 / 0.219 |
| Quay X | IMU + Từ kế | 424.2 | 424.4 | 241.9 | 47.1 | 4.13 | 1.20 | 0.045 / 0.036 / 0.029 |
| Quay Y | Chỉ IMU | 9641.4 | 9641.4 | 3862.9 | 1071.3 | -201.18 | 73.43 | 0.278 / 0.528 / 0.262 |
| Quay Y | IMU + Từ kế | 303.0 | 303.0 | 122.6 | 33.7 | -13.37 | 5.74 | 0.041 / 0.038 / 0.033 |
| Quay Z | Chỉ IMU | 2371.8 | 2371.8 | 1039.4 | 263.5 | 124.27 | 64.20 | 0.074 / 0.075 / 0.712 |
| Quay Z | IMU + Từ kế | 2215.2 | 2215.2 | 952.2 | 246.1 | 126.46 | 64.99 | 0.080 / 0.064 / 0.072 |
| Cố định | Chỉ IMU | 14213.3 | 14213.3 | 5649.6 | 1579.3 | 567.27 | 257.05 | 0.845 / 0.541 / 0.934 |
| Cố định | IMU + Từ kế | 2769.2 | 2791.3 | 1344.2 | 307.7 | 313.21 | 137.56 | 0.161 / 0.103 / 0.110 |
