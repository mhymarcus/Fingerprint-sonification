# 指紋圖像聲音化系統 — 使用者手冊（繁體中文）

這份手冊說明如何在自己的電腦上，把一張指紋相片轉換成聲音、獨立聲音層及幾何分析資料。

本系統在本機離線運行。它把可見的指紋幾何轉換成聲音，不會進行身份辨識、身份驗證或人口屬性分類。

## 1. 你需要準備甚麼？

最簡單的使用方式只需要：

- 一部安裝了 Python 3.11 或以上版本的電腦；
- 本專案；
- 一張指紋相片。

使用現有的 `cv` 分析模式時：

- 不需要 SOCOFing 或其他 dataset；
- 不需要訓練模型；
- 不需要 ML checkpoint；
- 不需要連接網絡。

### 建議相片規格

- 格式：JPG、JPEG、PNG 或 BMP；
- 相片中最好只有一隻手指；
- 指紋應佔相片大部分面積；
- 保持清晰對焦，避免手震；
- 使用均勻光線，避免強烈反光；
- 指紋與背景應有明顯對比；
- 建議保留未經社交媒體壓縮的原始相片。

HEIC/HEIF 暫時不能直接輸入。macOS 可先轉成 JPG：

```bash
sips -s format jpeg "/path/to/input.heic" --out "/path/to/fingerprint.jpg"
```

> 指紋是敏感的生物特徵資料。請確認你有權使用相片，並妥善保存原圖、分析結果和輸出資料。

## 2. 第一次安裝

在 Terminal 進入專案資料夾：

```bash
cd /path/to/Classic-Fingerprint-Pipeline
```

建立 Python 虛擬環境：

```bash
python3 -m venv venv
```

啟動虛擬環境。

macOS/Linux：

```bash
source venv/bin/activate
```

Windows PowerShell：

```powershell
venv\Scripts\Activate.ps1
```

安裝基本依賴：

```bash
python3 -m pip install -r requirements.txt
```

目前一般相片聲音化不需要安裝 `requirements-ml.txt`。

## 3. 最快開始方法

每次使用前，先進入專案並啟動虛擬環境：

```bash
cd /path/to/Classic-Fingerprint-Pipeline
source venv/bin/activate
```

執行分析：

```bash
python3 main.py \
  --input "/Users/visualarts/Downloads/marcus_art_project/Classic-Fingerprint-Pipeline/images/user_A_enroll.jpg" \
  --output "output/my_fingerprint.wav" \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug
```

請把 `/path/to/fingerprint.jpg` 換成實際相片位置。如果路徑包含空格，必須保留雙引號。

例如：

```bash
python3 main.py \
  --input "/Users/your-name/Downloads/my fingerprint.jpg" \
  --output "output/my_fingerprint.wav" \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug
```

如果同名輸出已經存在，而你確定要取代它，加入：

```text
--overwrite
```

完整例子：

```bash
python3 main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/my_fingerprint.wav" \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug \
  --overwrite
```

## 4. 播放及查看結果

macOS 播放最終混音：

```bash
open output/my_fingerprint.wav
```

查看幾何分析圖：

```bash
open output/my_fingerprint_analysis/geometry_analysis.png
```

你亦可以在 Finder、Windows Explorer、DAW 或 TouchDesigner 中開啟輸出檔案。

## 5. 輸出檔案說明

如果指定：

```text
--output output/my_fingerprint.wav
```

系統會產生：

```text
output/my_fingerprint.wav
output/my_fingerprint_ridge_length.wav
output/my_fingerprint_ridge_position.wav
output/my_fingerprint_ridge_orientation.wav
output/my_fingerprint_ridge_texture.wav
output/my_fingerprint_analysis/
├── geometry.json
├── ridge_segments.csv
├── geometry_analysis.png
├── stems_waveform.png
└── mix_spectrogram.png
```

### 音訊檔案

| 檔案 | 內容 |
| --- | --- |
| `my_fingerprint.wav` | 四個聲音層的最終混音 |
| `my_fingerprint_ridge_length.wav` | 脊線長度及可信度形成的持續音 |
| `my_fingerprint_ridge_position.wav` | 脊線水平及垂直位置形成的時間與音高事件 |
| `my_fingerprint_ridge_orientation.wav` | 整體方向場形成的連續音色變化 |
| `my_fingerprint_ridge_texture.wav` | 脊線間距與曲率形成的細節聲音 |

音訊預設為單聲道、16-bit PCM、44.1 kHz。所有 stems 使用相同時間軸，可在 DAW 或 TouchDesigner 中同步疊加。

### 分析檔案

| 檔案 | 用途 |
| --- | --- |
| `geometry_analysis.png` | 檢查原圖、增強結果、ROI、ridge mask、骨架及方向場 |
| `geometry.json` | 完整幾何線段及聲音事件資料 |
| `ridge_segments.csv` | 適合 Excel、Python、TouchDesigner 使用的線段表格 |
| `stems_waveform.png` | 比較最終混音和四個聲音層的時間分布 |
| `mix_spectrogram.png` | 查看最終聲音的頻譜變化 |

## 6. 如何判斷分析是否成功？

打開 `geometry_analysis.png`，檢查：

1. `Raw image` 是否顯示正確相片；
2. `Foreground ROI mask` 是否主要覆蓋手指；
3. `Ridge-level mask` 是否跟隨可見脊線，而不是整隻手指變成一大塊；
4. `Skeleton` 是否保留主要指紋線條；
5. `Orientation field` 是否跟隨指紋的環形、弓形或渦形方向。

陰影、皮膚摺痕、反光和背景紋理也可能被分析成線條。這些內容會成為聲音的一部分。如果只想分析指紋核心，可先裁剪相片再運行。

## 7. 常用選項

查看完整選項：

```bash
python main.py --help
```

| 選項 | 功能 | 例子 |
| --- | --- | --- |
| `--input` | 指定輸入相片 | `--input "photo.jpg"` |
| `--output` | 指定最終 WAV | `--output "output/result.wav"` |
| `--analysis` | 分析模式 | 建議使用 `--analysis cv` |
| `--duration` | 聲音長度，單位為秒 | `--duration 45` |
| `--export-stems` | 匯出四個獨立聲音層 | 無額外數值 |
| `--debug` | 匯出診斷圖 | 無額外數值 |
| `--overwrite` | 覆蓋同名輸出 | 無額外數值 |
| `--max-events` | 限制聲音事件數量 | `--max-events 100` |
| `--minimum-ridge-length` | 忽略過短線段 | `--minimum-ridge-length 6` |
| `--sample-rate` | 音訊取樣率 | `--sample-rate 44100` |

## 8. 調整聲音

四個層的增益選項：

```text
--length-gain
--position-gain
--orientation-gain
--texture-gain
--global-gain
```

例子：

```bash
python3 main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/softer_texture.wav" \
  --analysis cv \
  --duration 45 \
  --length-gain 0.8 \
  --position-gain 0.5 \
  --orientation-gain 0.7 \
  --texture-gain 0.25 \
  --global-gain 0.8 \
  --export-stems \
  --debug
```

關閉一個聲音層：

```bash
python3 main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/no_texture.wav" \
  --analysis cv \
  --disable-layer texture \
  --export-stems
```

可使用的層名稱：`length`、`position`、`orientation`、`texture`。如果要關閉多個層，可以重複使用 `--disable-layer`。

## 9. Dataset 和 ML 模式

### 一般使用者

把新相片轉成聲音時，請使用：

```text
--analysis cv
```

這個流程不需要 dataset。

### 研究或模型開發

`--analysis ml` 只適合已經有真實、完成訓練的 ridge-segmentation checkpoint 時使用：

```bash
python3 main.py \
  --input "/path/to/fingerprint.png" \
  --output "output/ml_result.wav" \
  --analysis ml \
  --checkpoint "checkpoints/ridge_segmentation.pt" \
  --export-stems
```

目前專案沒有可宣稱為已驗證的真實模型 checkpoint。SOCOFing 提供原始及 altered 指紋，但沒有 pixel-level ridge masks 或 orientation annotations，因此不能直接用來監督式訓練脊線分割模型。

請不要把未訓練的 U-Net 或合成資料測試結果當成真實模型結果。

## 10. TouchDesigner 使用方法

1. 在 TouchDesigner 建立 `Audio File In CHOP`；
2. 分別載入最終混音或四個 stems；
3. 因為 stems 使用相同時間軸，可以同步播放；
4. 可將不同層映射至視覺參數：
   - length → 大型形態、亮度或長時間變化；
   - position → 事件觸發及空間移動；
   - orientation → 旋轉、方向或色彩變形；
   - texture → 粒子、噪聲或細節密度。

系統目前不會自動建立 `.toe` 專案，也沒有即時相機或掃描器輸入。

## 11. 常見問題

### `python: command not found`

嘗試：

```bash
python3 main.py --help
```

並確認已安裝 Python 3.11 或以上。

### `ModuleNotFoundError`

通常是虛擬環境未啟動，或者依賴尚未安裝：

```bash
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 輸出檔案已存在

選擇新的輸出名稱，或者在確定要取代舊檔後加入：

```text
--overwrite
```

### 相片無法讀取

- 確認路徑正確；
- 路徑包含空格時使用雙引號；
- 確認格式為 JPG、JPEG、PNG 或 BMP；
- HEIC/HEIF 請先轉成 JPG。

### 分析圖與原圖差別很小

請查看 `geometry_analysis.png` 的 ridge mask、skeleton 和 orientation field，而不是只比較原圖與 enhanced grayscale。真正控制幾何聲音的是線段資料和方向場。

### 抽取了太多背景或摺痕

- 先裁剪至主要指紋區域；
- 改善光線和對焦後重新拍攝；
- 使用較大的 `--minimum-ridge-length` 過濾短碎線；
- 對結果有疑問時，以診斷圖作人工確認。

### Matplotlib cache 警告

可以指定暫存位置：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fingerprint-mpl python main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/result.wav" \
  --analysis cv \
  --debug
```

### 聲音太安靜或太大聲

使用 `--global-gain` 及各層 gain 調整。為避免 clipping，混音器會在需要時自動降低整體增益。

## 12. 測試安裝是否正常

執行測試：

```bash
python3 -m pytest -q
```

或先查看程式幫助：

```bash
python3 main.py --help
```

如果 `--help` 能正常顯示，而且測試通過，即表示基本安裝正常。

## 13. 建議工作流程

```text
保留原始相片
  → 複製一份作裁剪或格式轉換
  → 使用 CV 模式產生聲音及 debug 圖
  → 人工檢查 geometry_analysis.png
  → 比較四個 stems
  → 調整 layer gains 或裁剪範圍
  → 匯入 DAW / TouchDesigner 作進一步創作
```

相同相片和相同參數會產生相同結果，方便比較不同照片或聲音設定。
