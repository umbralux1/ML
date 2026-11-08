# Удаление модели Qwen2.5-7B-Instruct

Модель `Qwen/Qwen2.5-7B-Instruct` хранится в кэше Hugging Face и занимает около 15 ГБ:

```
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
```

Команды ниже выполняются из корня проекта.

## Через CLI Hugging Face

1. Посмотреть, что лежит в кэше:

   ```bash
   ./venv/bin/hf cache ls
   ```

2. Проверить, что будет удалено (без удаления):

   ```bash
   ./venv/bin/hf cache rm model/Qwen/Qwen2.5-7B-Instruct --dry-run
   ```

3. Удалить модель (команда попросит подтверждение, `-y` его отключает):

   ```bash
   ./venv/bin/hf cache rm model/Qwen/Qwen2.5-7B-Instruct
   ```

## Вручную

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
```

## После удаления

`main.py` при следующем запуске снова скачает веса, поэтому перед запуском нужно около 15 ГБ свободного места.
