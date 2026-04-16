from train_evaluate import run_training_pipeline


def train_and_save_model():
    """
    Backward-compatible entrypoint.
    Использует расширенный pipeline из train_evaluate.py
    (сравнение моделей + валидация по доменам + отчёты).
    """
    return run_training_pipeline()


if __name__ == "__main__":
    train_and_save_model()