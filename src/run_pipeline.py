import sys
import subprocess
from pathlib import Path

# --- БЛОК ПОИСКА КОРНЯ ПРОЕКТА ---
_current = Path(__file__).resolve()
ROOT_DIR = _current
while not (ROOT_DIR / "config.yaml").exists() and ROOT_DIR.parent != ROOT_DIR:
    ROOT_DIR = ROOT_DIR.parent
# ----------------------------------

sys.path.insert(0, str(ROOT_DIR))

LOGS_DIR = ROOT_DIR / "results" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def run_step(name, command, log_filename):
    print(f"\n{'=' * 60}")
    print(f" ШАГ: {name}")
    print(f"{'=' * 60}")

    log_path = LOGS_DIR / log_filename
    print(f"[log] Вывод сохраняется в: {log_path}")

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,  # <--- Всегда корень проекта
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Шаг '{name}' упал с кодом {e.returncode}. Смотри лог: {log_path}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[STOP] Прервано пользователем.")
        sys.exit(0)

    print(f"[ok] Шаг '{name}' завершен. Лог: {log_path.name}")


def main():
    print(f"🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА (корень: {ROOT_DIR})")

    # Шаг 1: Leave-One-Out подбор порогов
    run_step(
        "Кросс-валидация (LOO)",
        [sys.executable, "src/scripts/cv_threshold.py"],
        "01_loo.txt"
    )

    # Шаг 2: Обновление порогов
    print("\n⚠️  ВАЖНО: Возьмите средние пороги из лога выше и обновите config.yaml")
    print("   (например, если LOO выдал 'Средний порог: 0.65', поставьте hsv: 0.65)")
    print(f"   Лог LOO: {LOGS_DIR / '01_loo.txt'}")
    input("Нажмите Enter, чтобы продолжить...")

    # Шаг 3: Генерация предсказаний
    run_step(
        "Генерация предсказаний (main.py)",
        [sys.executable, "src/main.py"],
        "02_predictions.txt"
    )

    # Шаг 4: Генерация baseline'ов
    run_step(
        "Генерация baseline'ов",
        [sys.executable, "src/matchs/baselines.py"],
        "03_baselines.txt"
    )

    # Шаг 5: Оценка качества
    run_step(
        "Оценка качества (ev_all.py)",
        [sys.executable, "src/matchs/ev_all.py"],
        "04_metrics.txt"
    )

    # Шаг 6: Построение графиков
    run_step(
        "Построение графиков",
        [sys.executable, "src/plot_run.py"],
        "05_plots.txt"
    )

    print("\nПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
    print(f"\nВсе логи в: {LOGS_DIR}")
    print(f"Все результаты в: {ROOT_DIR / 'results'}")


if __name__ == "__main__":
    main()