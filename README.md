# Video Content Ranking — Scene Segmentation (HSV)

Пайплайн сегментации видео на сцены: детекция шотов → HSV-подписи шотов → группировка в сцены по сходству гистограмм.

## Что делает

Для каждого видео из `data/raw/`:
1. **PySceneDetect** находит шоты (по содержимому, не по времени).
2. Для каждого шота считается **HSV-гистограмма** (усреднение по кадрам, взятым каждые N кадров).
3. Шоты **группируются в сцены** по сходству гистограмм: если новый шот похож на один из последних в текущей группе — присоединяется, иначе группа закрывается (при условии, что её длительность ≥ `min_duration`).
4. Результат сохраняется в `data/result_hist/<имя>_data.csv` и `<имя>_data.npy`.

## Требования

- Python 3.10+
- Зависимости из `requirements.txt`

## Установка

```bash
pip install -r requirements.txt
```

Если хотите изолированное окружение:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Подготовка данных

Положить видеофайлы в `data/raw/`. Поддерживаются расширения:
`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`.

Пример структуры:

```
.
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── movie_1.mp4
│   │   ├── movie_2.mkv
│   │   └── ...
│   └── result_hist/         ← создастся автоматически
└── src/
    └── main.py
```

## Запуск

**Из корня проекта** (там, где лежат `config.yaml`, `data/`, `src/`):

```bash
python src/main.py
```

## Параметры CLI

Все параметры опциональны. Если не указаны — берутся из `config.yaml`.

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--config` | `config.yaml` | путь к конфигу |
| `--raw` | из конфига | папка с исходными видео |
| `--out` | из конфига | папка для результатов |
| `--threshold` | 0.6 | порог сходства гистограмм (0..1) |
| `--lookback` | 5 | сколько последних шотов группы сравнивать |
| `--min-duration` | 2.0 | минимальная длительность сцены (сек) |
| `--skip-existing` | off | пропускать уже обработанные файлы |
| `--limit` | — | обработать только первые N файлов |

CLI-параметры **перекрывают** значения из конфига.

### Примеры

```bash
# стандартный запуск
python src/main.py

# тест на первых 2 файлах
python src/main.py --limit 2

# другой порог, пропустить уже готовые
python src/main.py --threshold 0.5 --skip-existing

# указать свои папки
python src/main.py --raw data/test --out data/test_out
```

## Конфиг

Все настройки — в `config.yaml` в корне проекта:

```yaml
paths:
  raw: data/raw
  out: data/result_hist

video_exts:
  - .mp4
  - .mkv
  - .avi
  - .mov
  - .webm
  - .flv

signature:
  scale: 0.5           # уменьшение кадра перед HSV
  frame_step: 5        # брать каждый N-й кадр внутри шота
  h_bins: 8            # бинов по Hue (0..180 в OpenCV)
  s_bins: 8
  v_bins: 8

grouping:
  threshold: 0.6       # порог сходства гистограмм
  lookback: 5          # окно сравнения
  min_duration: 2.0    # минимальная длительность сцены, сек

logging:
  level: INFO
  file: null           # путь к файлу лога, если нужно писать в файл
```

## Формат вывода

### CSV (`data/result_hist/<имя>_data.csv`)

| Поле | Описание |
|---|---|
| `scene_id` | номер сцены (с 0) |
| `start_time` | начало сцены, сек |
| `end_time` | конец сцены, сек |
| `cuts` | количество шотов внутри сцены |

Пример:

```csv
scene_id,start_time,end_time,cuts
0,0.000,5.800,2
1,5.800,15.300,4
2,15.300,22.100,3
```


## Структура проекта

```
.
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                        ← входные видео
│   └── result_hist/                ← результаты
└── src/
    ├── main.py                     ← запускатор
    ├── scene_detect.py             ← PySceneDetect
    ├── histogram_method/
    │   ├── cut_histogrm.py         ← HSV-подписи шотов
    │   └── group_by_hist.py        ← группировка шотов в сцены
    └── utils/
        ├── config.py               ← загрузка config.yaml
        └── io_util.py              ← сохранение CSV / NPY
```

