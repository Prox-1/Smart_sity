# bot.py
import html
import os
import asyncio
from functools import partial
from typing import Set, Optional

import requests

from telegram import (
    Update,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    constants,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    Defaults,
    filters,
)

# =========================
# Настройки и константы
# =========================

BOT_TOKEN = os.environ.get(
    "TG_BOT_TOKEN", "8473383190:AAEObZeIWhRW6sDsGu3AW56eTbzBuVAUB6I")
SIM_API = os.environ.get("SIM_API", "http://127.0.0.1:8081")

TRUSTED_USER_IDS: Set[int] = {
    1564311227, 5044597738
}

EMOJI = {
    "ok": "✅",
    "fail": "❌",
    "gear": "⚙️",
    "health": "🩺",
    "spawn": "🚧",
    "clear": "🧹",
    "lock": "🔒",
    "geo": "📍",
    "menu": "📋",
    "info": "ℹ️",
}

# =========================
# Утилиты
# =========================


def safe_html(obj: object) -> str:
    return html.escape(str(obj), quote=False)


def is_trusted(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return uid in TRUSTED_USER_IDS


async def _run_in_thread(func, *args, **kwargs):
    return await asyncio.to_thread(partial(func, *args, **kwargs))


async def http_get(url: str, **kwargs) -> requests.Response:
    # Неблокирующий вызов requests в отдельном потоке
    return await _run_in_thread(requests.get, url, **kwargs)


async def http_post(url: str, **kwargs) -> requests.Response:
    return await _run_in_thread(requests.post, url, **kwargs)


def build_location_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton("Отправить геопозицию", request_location=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Нажмите кнопку, чтобы отправить локацию",
        is_persistent=False,
    )


def build_inline_menu(trusted: bool) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(
            f"{EMOJI['health']} Проверка связи", callback_data="health")],
    ]
    if trusted:
        rows.append(
            [InlineKeyboardButton(
                f"{EMOJI['geo']} Авария по геопозиции", callback_data="spawn_here")]
        )
        rows.append(
            [InlineKeyboardButton(
                f"{EMOJI['clear']} Очистить все…", callback_data="clear_all_prompt")]
        )
    rows.append([InlineKeyboardButton("❓ Помощь", callback_data="help_open")])
    return InlineKeyboardMarkup(rows)


def help_text(trusted: bool) -> str:
    return (
        "<b>Справка</b>\n"
        f"{EMOJI['info']} Бот для управления авариями в симуляции SUMO.\n\n"
        "<b>Команды</b>\n"
        "• /start — главное меню\n"
        "• /whoami — показать ваш user_id\n"
        "• /health — проверка связи с симуляцией\n"
        "• /send_location_button — показать кнопку для отправки геопозиции\n"
        + (
            "\n<b>Для доверенных</b>\n"
            "• /spawn_here — затем отправьте геопозицию сообщением Location\n"
            "• /clear_all\n"
            if trusted else f"\n{EMOJI['lock']} Доступ к командам управления авариями ограничён."
        )
    )


def home_text() -> str:
    return (
        f"<b>{EMOJI['menu']} Главное меню</b>\n"
        "• Используйте кнопки ниже для быстрых действий.\n"
        "• Команды поддерживают HTML-форматирование и короткие подсказки.\n"
        "• Для аварии по геопозиции — нажмите кнопку, затем отправьте Location."
    )


async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    trusted = is_trusted(update)
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING
    )
    await (update.effective_message or update.effective_chat).reply_text(
        home_text(),
        reply_markup=build_inline_menu(trusted),
    )

# =========================
# post_init: красивое меню команд
# =========================


async def post_init(app: Application):
    commands = [
        BotCommand("start", "Запуск и главное меню"),
        BotCommand("help", "Справка и примеры"),
        BotCommand("whoami", "Показать ваш user_id"),
        BotCommand("health", "Проверка связи с симуляцией"),
        BotCommand("send_location_button",
                   "Показать кнопку для отправки геопозиции"),
        # Ниже — только валидные имена, примеры в описаниях
        BotCommand("spawn_here", "После команды отправьте свою геопозицию"),
        BotCommand("clear_all", "Удалить все аварии (для доверенных)"),
    ]
    await app.bot.set_my_commands(commands)

# =========================
# Хендлеры команд
# =========================


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_main_menu(update, context)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(help_text(is_trusted(update)))


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"<b>Ваш user_id:</b> <code>{update.effective_user.id if update.effective_user else 'unknown'}</code>"
    )


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)
    try:
        r = await http_get(f"{SIM_API}/api/health", timeout=5)
        ok = r.ok and r.json().get("ok") is True
        icon = EMOJI["ok"] if ok else EMOJI["fail"]
        await update.message.reply_text(f"{EMOJI['health']} Симуляция: {icon} {'OK' if ok else 'нет'}")
    except Exception as e:
        await update.message.reply_text(f"{EMOJI['fail']} Симуляция недоступна: <code>{safe_html(e)}</code>")


async def spawn_here_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_trusted(update):
        await update.message.reply_text(f"{EMOJI['lock']} Доступ запрещён.")
        return
    await update.message.reply_text(
        "Отправьте вашу геопозицию (вложение Location), затем я размещу аварию на ближайшей полосе.",
        reply_markup=build_location_keyboard(),
    )


async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_trusted(update):
        await update.message.reply_text(f"{EMOJI['lock']} Доступ запрещён.")
        return
    if not update.message or not update.message.location:
        return

    lon = update.message.location.longitude
    lat = update.message.location.latitude

    await context.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)
    try:
        payload = {"lon": lon, "lat": lat}
        r = await http_post(f"{SIM_API}/api/spawn_geo", json=payload, timeout=7)
        if r.ok and r.json().get("ok"):
            await update.message.reply_text(
                f"{EMOJI['spawn']} Запрос на аварию по геопозиции:\n"
                f"lon=<code>{lon:.6f}</code>, lat=<code>{lat:.6f}</code>",
                reply_markup=ReplyKeyboardRemove(),
            )
        else:
            await update.message.reply_text(f"{EMOJI['fail']} Ошибка: <code>{safe_html(r.text)}</code>")
    except Exception as e:
        await update.message.reply_text(f"{EMOJI['fail']} Сбой запроса: <code>{safe_html(e)}</code>")


async def clear_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_trusted(update):
        await update.message.reply_text(f"{EMOJI['lock']} Доступ запрещён.")
        return

    await context.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)
    try:
        r = await http_post(f"{SIM_API}/api/clear_all", json={}, timeout=10)
        if r.ok and r.json().get("ok"):
            await update.message.reply_text(f"{EMOJI['clear']} Удаление всех аварий запрошено")
        else:
            await update.message.reply_text(f"{EMOJI['fail']} Ошибка: <code>{safe_html(r.text)}</code>")
    except Exception as e:
        await update.message.reply_text(f"{EMOJI['fail']} Сбой запроса: <code>{safe_html(e)}</code>")


async def send_location_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_trusted(update):
        await update.message.reply_text(f"{EMOJI['lock']} Доступ запрещён.")
        return

    kb = build_location_keyboard()
    await update.message.reply_text(
        "Нажмите кнопку, чтобы отправить геопозицию:",
        reply_markup=kb,
    )

# =========================
# Inline-кнопки (callback_data)
# =========================


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    trusted = is_trusted(update)
    await q.answer()

    if data == "health":
        try:
            r = await http_get(f"{SIM_API}/api/health", timeout=5)
            ok = r.ok and r.json().get("ok") is True
            icon = EMOJI["ok"] if ok else EMOJI["fail"]
            await q.edit_message_text(
                f"{EMOJI['health']} Симуляция: {icon} {'OK' if ok else 'нет'}",
                reply_markup=build_inline_menu(trusted)
            )
        except Exception as e:
            await q.edit_message_text(
                f"{EMOJI['fail']} Симуляция недоступна: <code>{safe_html(e)}</code>",
                reply_markup=build_inline_menu(trusted)
            )
    elif data == "clear_all_confirm":
        if not trusted:
            return await q.answer("Доступ запрещён.", show_alert=True)
        try:
            r = await http_post(f"{SIM_API}/api/clear_all", json={}, timeout=10)
            if r.ok and r.json().get("ok"):
                await q.edit_message_text(
                    f"{EMOJI['clear']} Все аварии будут удалены.",
                    reply_markup=build_inline_menu(trusted)
                )
            else:
                await q.edit_message_text(
                    f"{EMOJI['fail']} Ошибка: <code>{safe_html(r.text)}</code>",
                    reply_markup=build_inline_menu(trusted)
                )
        except Exception as e:
            await q.edit_message_text(
                f"{EMOJI['fail']} Сбой запроса: <code>{safe_html(e)}</code>",
                reply_markup=build_inline_menu(trusted)
            )
    elif data == "spawn_here":
        if not trusted:
            return await q.answer("Доступ запрещён.", show_alert=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Отправьте вашу геопозицию сообщением Location:",
            reply_markup=build_location_keyboard(),
        )

    elif data == "clear_all_prompt":
        if not trusted:
            return await q.answer("Доступ запрещён.", show_alert=True)
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Да, очистить", callback_data="clear_all_confirm"),
                InlineKeyboardButton("Отмена", callback_data="menu"),
            ]
        ])
        await q.edit_message_text(f"{EMOJI['clear']} Очистить все аварии?", reply_markup=kb)
    elif data == "help_open":
        await q.edit_message_text(help_text(trusted), reply_markup=build_inline_menu(trusted))

    elif data == "menu":
        await q.edit_message_text(home_text(), reply_markup=build_inline_menu(trusted))

# =========================
# Error handler
# =========================


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Минимальный лог: можно заменить на logging
    print("Error:", context.error)

# =========================
# main
# =========================


def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "TG_BOT_TOKEN не задан. Установите переменную окружения TG_BOT_TOKEN.")

    defaults = Defaults(
        parse_mode=constants.ParseMode.HTML,
    )

    app = (
        Application
        .builder()
        .token(BOT_TOKEN)
        .defaults(defaults)
        .build()
    )

    app.post_init = post_init

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("health", health))

    app.add_handler(CommandHandler("spawn_here", spawn_here_prompt))
    app.add_handler(CommandHandler("clear_all", clear_all))

    app.add_handler(CommandHandler(
        "send_location_button", send_location_button))

    # Сообщения
    app.add_handler(MessageHandler(filters.LOCATION, location_handler))

    # Inline-кнопки
    app.add_handler(CallbackQueryHandler(on_button))

    # Ошибки
    app.add_error_handler(error_handler)

    app.run_polling()


if __name__ == "__main__":
    main()
