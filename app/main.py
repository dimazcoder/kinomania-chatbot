import sys

from app.services.ai_service import load_model, train_model
from app.services.bot_service import start_bot


def main():
    print("Starting bot...")
    model = load_model()
    start_bot(model)


def train():
    print('Training...')
    train_model()


if __name__ == '__main__':
    if len( sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    else:
        main()