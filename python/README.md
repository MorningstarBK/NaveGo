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
| `sim/gnss.py` | GNSS vị trí/vận tốc, có khoảng mất tín hiệu | `gnss_gen.m`, `gnss_outage.m` |
| `fusion.py` | EKF trạng thái lỗi 16 trạng thái, vòng kín | `ins-gnss/ins_gnss.m` |
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
