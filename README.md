# VisualSegmentationTrial

Эксперимент по **визуальной сегментации сцен в кино**: сравнение трёх методов подписи шотов (HSV-гистограмма, CLIP, DINOv2) на задаче группировки шотов в сцены.

## Идея

Видео режется на **шоты** (PySceneDetect), каждый шот описывается **вектором-подписью**, затем шоты **группируются в сцены** по сходству подписей. Сравниваются три подписи:

| Метод | Признак | Размер вектора | Метрика сходства |
|---|---|---|---|
| **HSV** | 3D-гистограмма (H×S×V) | 512 | пересечение гистограмм |
| **CLIP** | эмбеддинг ViT-B/32 (OpenAI) | 512 | косинус |
| **DINOv2** | эмбеддинг ViT-B/14 (Meta) | 768 | косинус |

Результат каждого метода сравнивается с **ручной разметкой** через precision / recall / F1 по границам сцен с допуском ±2 сек.

## Что делает пайплайн

Для каждого видео из `data/raw/`:

1. **PySceneDetect** (`ContentDetector`) находит шоты.
2. Для каждого шота считается **подпись** (HSV / CLIP / DINOv2).
3. Шоты **группируются в сцены**:
   - идём по шотам слева направо;
   - новый шот сравнивается с последними `lookback` шотами текущей группы;
   - если максимум сходства ≥ порога — присоединяется;
   - иначе группа закрывается (при условии, что её длительность ≥ `min_duration`).
4. Для **HSV** порог фиксированный; для **CLIP/DINOv2** — **адаптивный** (`mean − k·std` по парам соседних шотов).
5. Результат сохраняется в `data/result_<method>/<имя>_data.csv`.

## Требования

- Python 3.10+
- Зависимости из `requirements.txt`

## Установка

```bash
pip install -r requirements.txt
```

Изолированное окружение:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

**Первый запуск CLIP/DINOv2** скачает веса (~350 МБ каждая) из HuggingFace в `~/.cache/huggingface/`. Нужен интернет.

## Подготовка данных

1. Видеофрагменты → `data/raw/` (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`).
2. Ручная разметка → `data/ground_truth/<имя>.csv`:

```csv
scene_id,start_sec,end_sec
0,0.0,7.0
1,7.0,53.0
```

Имя файла разметки должно **совпадать** с именем видео (без расширения).

Структура:

```
.
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── movie_1.mp4
│   │   └── movie_2.mkv
│   ├── ground_truth/
│   │   ├── movie_1.csv
│   │   └── movie_2.csv
│   ├── result_hsv/          ← создаётся автоматически
│   ├── result_clip/
│   └── result_dino/
└── src/
    └── main.py
```

## Запуск

**Из корня проекта** (там, где `config.yaml`, `data/`, `src/`):

```bash
# все три метода, все видео
python src/main.py

# только HSV (быстро, без скачивания моделей)
python src/main.py --methods hsv

# только CLIP
python src/main.py --methods clip

# только DINOv2 (медленно на CPU)
python src/main.py --methods dino --limit 1

# тест на первых 2 файлах
python src/main.py --limit 2

# пропустить уже обработанные
python src/main.py --skip-existing
```

## Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--config` | `config.yaml` | путь к конфигу |
| `--methods` | `hsv clip dino` | какие методы прогнать |
| `--raw` | из конфига | папка с исходными видео |
| `--threshold` | — | override порога для всех методов |
| `--lookback` | из конфига | сколько последних шотов группы сравнивать |
| `--min-duration` | из конфига | минимальная длительность сцены, сек |
| `--skip-existing` | off | пропускать уже обработанные |
| `--limit` | — | обработать только первые N файлов |

CLI-параметры **перекрывают** значения из конфига.

## Конфиг

Все настройки — в `config.yaml`:

```yaml
paths:
  raw: data/raw

video_exts: [.mp4, .mkv, .avi, .mov, .webm, .flv]

signature:
  hsv:
    scale: 0.5          # уменьшение кадра перед HSV
    frame_step: 5       # каждый N-й кадр внутри шота
    h_bins: 8
    s_bins: 8
    v_bins: 8
  clip:
    frame_step: 30
  dino:
    frame_step: 60      # DINOv2 без GPU — медленный; 60 = 1 кадр в 2 сек

grouping:
  threshold:
    hsv: 0.6            # фиксированный для HSV
    clip: 0.75          # не используется, если threshold=None
    dino: 0.75
  lookback: 5
  min_duration: 5.0
  k: 1.0                # коэффициент адаптивного порога для CLIP/DINO

logging:
  level: INFO
  file: null
```

## Формат вывода

### Результаты сегментации — `data/result_<method>/<имя>_data.csv`

```csv
scene_id,start_time,end_time,cuts
0,0.0,7.147,2
1,7.147,52.705,3
```

| Поле | Описание |
|---|---|
| `scene_id` | номер сцены (с 0) |
| `start_time`, `end_time` | границы сцены, сек |
| `cuts` | количество шотов внутри сцены |

### Оценка

```bash
python src/matches/ev_all.py
```

Скрипт **автоматически находит** все папки `data/result_*/`, сравнивает их с `data/ground_truth/` и печатает таблицу:

```
video                    |       clip        |       dino        |        hsv
---------------------------------------------------------------------------
inglouriosBasterds       | P=… R=… F1=…      | P=… R=… F1=…      | P=… R=… F1=…
pulpFiction              | P=… R=… F1=…      | P=… R=… F1=…      | P=… R=… F1=…
...
---------------------------------------------------------------------------
СРЕДНЕЕ                  | P=… R=… F1=…      | P=… R=… F1=…      | P=… R=… F1=…
```

## Методика оценки

- **Ground truth** — ручная разметка сцен, `data/ground_truth/<имя>.csv`. Точность ±1–2 сек, покрывается допуском.
- **Prediction** — `data/result_<method>/<имя>_data.csv`.
- **Границы сцен** — время `end_time` всех сцен, кроме последней (последняя — конец видео).
- **Сопоставление** — жадное: каждая предсказанная граница ищет ближайшую истинную в пределах допуска ±2 сек, один-к-одному. Логика в `src/matches/matches_bound.py` (класс `SceneEvaluator`).
- **Precision** = TP / (TP + FP), **Recall** = TP / (TP + FN), **F1** = 2·P·R / (P + R).

## Описание эксперимента

**Данные:** 4 отрывка из фильмов, длительностью ~1–4 минуты.

| Отрывок | Длительность | Сцен (ручная разметка) |
|---|---|---|
| Inglourious Basterds | ~53 сек | 2 |
| Pulp Fiction | ~228 сек | 4 |
| Snatch | ~137 сек | 3 |
| Requiem for a Dream | ~125 сек | 4 |

**Методы:** HSV-гистограмма, CLIP ViT-B/32, DINOv2-base.

**Метрика:** precision / recall / F1 границ сцен с допуском ±2 сек.

**Гиперпараметры:** для HSV — фиксированный порог 0.6; для CLIP/DINOv2 — адаптивный (`mean − k·std`, `k=1.0`); `lookback=5`, `min_duration=5.0`.

## Структура проекта

```
.
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                        ← входные видео (в .gitignore)
│   ├── ground_truth/               ← ручная разметка
│   ├── result_hsv/                 ← результаты HSV
│   ├── result_clip/                ← результаты CLIP
│   └── result_dino/                ← результаты DINOv2
└── src/
    ├── main.py                     ← запускатор пайплайна
    ├── scene_detect.py             ← PySceneDetect
    ├── histogram_method/
    │   ├── cut_histogrm.py         ← HSV-подписи шотов
    │   ├── clip_signature.py       ← CLIP-подписи
    │   ├── dino_signature.py       ← DINOv2-подписи
    │   ├── shot_signature.py       ← единый интерфейс
    │   ├── similarity.py           ← метрики сходства
    │   └── group_by_hist.py        ← группировка шотов в сцены
    ├── matches/
    │   ├── ev_all.py               ← сводная оценка по всем методам
    │   └── matches_bound.py        ← SceneEvaluator
    └── utils/
        ├── config.py               ← загрузка config.yaml
        └── io_util.py              ← сохранение CSV
```
