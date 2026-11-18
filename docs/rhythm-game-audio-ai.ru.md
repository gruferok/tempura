# Python-аудиовывод и AI-генерация карт для ритм-игры

**Дата проверки: 15 ноября 2026 года, UTC.**

Исследование двух задач: точное воспроизведение музыки в Python-ритм-игре и открытые проекты генерации карт 2023–2026 годов, прежде всего osu!mania / osu!taiko и другие дискретные дорожечные режимы.

**Методика и границы:** изучены официальные API, исходники Python-обёрток, GitHub README / лицензии / Releases и первичные статьи. Реальные задержки на аудиоустройстве не измерялись; модели не устанавливались, веса не скачивались, качество сгенерированных карт не проверялось прослушиванием и игрой. Упоминание checkpoint означает наличие файла или инструкции/ссылки у автора, а не подтверждение успешного inference. Сведения о готовности и лицензиях относятся к состоянию на указанную дату.

## 1. Короткий вывод

1. **Для нового Python-прототипа — `python-sounddevice` и callback `OutputStream`.** Основное преимущество здесь не обещанная «самая низкая задержка», а доступ к `outputBufferDacTime` и совместимым часам `Stream.time`. `PyAudio` предоставляет практически ту же временную основу через PortAudio и является полноценной альтернативой. [1](https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html) [2](https://people.csail.mit.edu/hubert/pyaudio/docs/)
2. **`pygame.mixer.music.get_pos()` не стоит делать главным источником времени для строгого judgment.** Это elapsed playback time в миллисекундах, без учёта стартового offset, а не документированный аппаратный DAC cursor. Сам pygame можно оставить для окна, графики и ввода. [3](https://www.pygame.org/docs/ref/music.html)
3. **Python `miniaudio` удобен для декодирования и воспроизведения, но его публичный `PlaybackDevice` не даёт аналога PortAudio DAC timestamps.** Возможности C-библиотеки miniaudio нельзя автоматически приписывать Python binding. [4](https://github.com/irmen/pyminiaudio/blob/e84ad4c/miniaudio.py)
4. **Первый практический эксперимент с картами — Mapperatorinator:** современный event Transformer, есть mania и taiko, публичные инструкции генерации. **MuG Diffusion** особенно релевантен для 4K, но его условия использования весов/выходных карт противоречивы. Для научных baseline полезны **AutoOsu**, **Beat-Aligned Spectrogram-to-Sequence**, **GenéLive!**, а для соседнего четырёхдорожечного режима — **ITGPT**. [5](https://github.com/OliBomby/Mapperatorinator) [6](https://github.com/Keytoyze/Mug-Diffusion) [7](https://github.com/issyun/AutoOsu) [8](https://github.com/stet-stet/goct_ismir2023) [9](https://github.com/KLab/AAAI-23.6040) [10](https://github.com/miguelomalley/ITGPT)
5. **`librosa`, `torchaudio` и `madmom` — не три конкурирующие модели lane assignment.** Первые две в основном дают инструменты обработки и признаки; madmom дополнительно поставляет обученные onset/beat-модели. Полноценная карта требует отдельного решения о плотности, дорожках, типах действий, длинных нотах и играбельности.
6. **Универсальный рейтинг «Diffusion > Transformer > CRNN» источниками не подтверждается.** Важнее целевой режим, данные, beat conditioning, представление событий, ограничения декодера и отдельная оценка тайминга и раскладки.

## 2. Что именно нужно измерять в аудиосистеме

Нельзя смешивать четыре разные величины:

- **Позиция чтения/декодирования:** до какого места файл уже превращён в PCM.
- **Позиция записи:** сколько frames приложение передало выходному потоку.
- **Оценка позиции вывода:** какие frames сейчас должны доходить до DAC по временной информации backend.
- **Фактически слышимый момент:** с учётом неизвестных драйверу очередей, внешнего устройства, беспроводного соединения и акустического тракта.

Для игрового song clock нужна третья величина с калибровкой остаточной ошибки четвёртой. Счётчик отправленных frames опережает звук на величину очереди. `time.perf_counter() - start_time` сам по себе не знает ни момента реального старта, ни состояния аудиобуфера.

**Маленькая задержка и стабильные часы — разные свойства.** Можно точно синхронизировать визуал с заранее буферизованной музыкой, но иметь неприятную задержку реактивных hitsounds. И наоборот, маленький буфер не исправляет неверно вычисленный playback cursor. PortAudio отдельно описывает scheduling jitter, `currentTime`, `outputBufferDacTime` и неизвестную часть latency. [11](https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines)

## 3. Сравнение обязательных аудиобиблиотек

| Библиотека | Что доступно для позиции | Управление задержкой | Сильные стороны | Вывод для ритм-игры |
|---|---|---|---|---|
| **pygame.mixer** | У `music` есть `get_pos()` в ms. Стартовый offset не включён. Публичного DAC timestamp нет | `mixer.pre_init(..., buffer=...)`; уменьшение буфера повышает риск dropout | Простые музыка, каналы и эффекты; интеграция с pygame | Хорош для быстрого игрового прототипа; не мой выбор для строгих аудиочасов |
| **python-sounddevice** | Callback: `time.outputBufferDacTime`, `time.currentTime`; снаружи — `stream.time`, в той же временной системе | `latency='low'` / численный запрос, `blocksize`, выбор устройства/backend; WASAPI/CoreAudio/ASIO-specific настройки | Удобный NumPy API, также `RawOutputStream`; хорошая документация callback и clock | **Рекомендуемый Python API**, если реализовать собственный transport и PCM-подачу |
| **PyAudio** | Callback: `time_info['output_buffer_dac_time']`, `current_time`; снаружи — `stream.get_time()` | `frames_per_buffer`, выбор устройства/host API, доступ к reported output latency | Та же PortAudio-основа, зрелый callback API | **Равноценная по принципу часов альтернатива**; нет оснований считать её заведомо медленнее sounddevice |
| **miniaudio / pyminiaudio** | Генератор получает необходимое число frames. В публичном Python `PlaybackDevice` нет presentation/DAC timestamp и метода позиции слышимого потока | `buffersize_msec`, `callback_periods`, backend, приоритет потока | Встроенные декодеры; компактное решение для PCM и проигрывания | Подходит для простого playback/декодирования. Для строгого clock понадобится дополнительный native/backend слой |

Источники таблицы: API pygame music/mixer, sounddevice, PyAudio и проверенный исходник pyminiaudio. [1](https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html) [2](https://people.csail.mit.edu/hubert/pyaudio/docs/) [3](https://www.pygame.org/docs/ref/music.html) [4](https://github.com/irmen/pyminiaudio/blob/e84ad4c/miniaudio.py) [12](https://www.pygame.org/docs/ref/mixer.html)

### 3.1. Существенные оговорки

**pygame.mixer**

- Документация определяет `get_pos()` как число миллисекунд, в течение которых музыка воспроизводилась; для абсолютной позиции стартовый offset нужно учитывать самостоятельно.
- Для MP3 точный старт может нарушаться из-за VBR и ID3. Нельзя обещать sample-accurate seeking только потому, что функция принимает дробное число секунд.
- Маленький `buffer` уменьшает одну часть latency, но не меняет семантику `get_pos()`. [3](https://www.pygame.org/docs/ref/music.html) [12](https://www.pygame.org/docs/ref/mixer.html)

**sounddevice / PyAudio**

- Оба используют **PortAudio**. При одинаковом host API, формате, устройстве и буферах нельзя заранее объявить один wrapper быстрее другого.
- `outputBufferDacTime` — временная метка первого frame текущего выходного буфера. Это информация/оценка backend, а не универсальная гарантия наблюдения физического DAC.
- В sounddevice `Stream.time` монотонен, имеет неопределённое начало отсчёта и **продолжает идти при остановленном stream**. Это часы, а не автоматически готовый song position.
- `latency='low'` — запрос. Реальная конфигурация может отличаться; reported latency тоже ограничена знаниями backend. Начать стоит с `blocksize=0`: жёстко заданный blocksize на некоторых host API требует дополнительной адаптационной буферизации.
- В стандартном PyAudio документирован `get_output_latency()`, а sounddevice дополнительно предоставляет удобные Python-классы platform-specific настроек. Наличие конкретного backend всё равно зависит от сборки PortAudio. [1](https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html) [2](https://people.csail.mit.edu/hubert/pyaudio/docs/) [11](https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines) [13](https://python-sounddevice.readthedocs.io/en/0.5.6/api/platform-specific-settings.html)

**miniaudio**

Проверены исходник default branch и содержимое исходного архива PyPI **1.73**, без установки и исполнения. У `PlaybackDevice` публичные методы не предоставляют DAC clock. В коде `buffersize_msec` по умолчанию равен **200** и передаётся в `periodSizeInMilliseconds`. Это **параметр запроса периода, не измеренная end-to-end задержка**. Настройки по умолчанию не следует считать low-latency конфигурацией. Позиция декодера/генератора не равна позиции звучания. [4](https://github.com/irmen/pyminiaudio/blob/e84ad4c/miniaudio.py) [14](https://pypi.org/project/miniaudio/1.73/)

### 3.2. Версии на дату проверки

Это snapshot PyPI, а не утверждение, что все перечисленные версии уже протестированы вместе:

| Пакет | Последний стабильный PyPI-релиз | Дата публикации |
|---|---|---|
| pygame | 2.6.1 | 2024-09-29 |
| sounddevice | 0.5.6 | 2026-08-17 |
| PyAudio | 0.2.14 | 2023-11-07 |
| miniaudio | 1.73 | 2026-11-10 |

Источники: метаданные PyPI для [15](https://pypi.org/pypi/pygame/json), [16](https://pypi.org/pypi/sounddevice/json), [17](https://pypi.org/pypi/PyAudio/json), [18](https://pypi.org/pypi/miniaudio/json). `pygame-ce` — отдельный проект: его issue/версии нельзя автоматически переносить на обычный pygame.

## 4. Как получить устойчивый song clock

Ниже — инженерная схема, выведенная из контракта PortAudio, **не готовая протестированная реализация и не обещание нулевой ошибки**.

### 4.1. Якорь «сэмпл песни ↔ время DAC»

Для непрерывного участка без pause/seek и с обычной скоростью:

1. Заранее декодировать аудио и согласовать PCM sample rate с выходным потоком.
2. В callback знать индекс `F0` первого **song PCM frame** записываемого буфера.
3. Сохранить вместе с ним `T0 = time.outputBufferDacTime`.
4. В игровом потоке прочитать `T = stream.time` и оценить:

```text
song_position_seconds = F0 / output_sample_rate + (T - T0)
```

В PyAudio используются те же данные с именами `output_buffer_dac_time` и `get_time()`.

**Почему это лучше подсчёта callbacks:** позиция интерполируется по аудиочасам и привязана к presentation time буфера, а не ступенчато меняется только при вызове Python-функции. Не нужно накапливать `delta time` игровых кадров или считать момент вызова callback моментом звучания. Однако неточные timestamps драйвера такая схема не исправляет. Контракт часов описан в API и timing guide. [1](https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html) [2](https://people.csail.mit.edu/hubert/pyaudio/docs/) [11](https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines)

### 4.2. Что обязательно добавить к формуле

- **Не вычитать output latency второй раз:** DAC anchor уже учитывает известное backend время до вывода.
- **Хранить timeline segments/epochs.** После seek, смены скорости, pause/resume или underrun старый линейный участок может быть недействителен. Если новый буфер ещё стоит в очереди, его якорь нельзя слепо применять к уже слышимому старому участку. Нужна история сегментов и их presentation boundaries.
- **Согласовать единицы.** Если `F0` относится к исходнику 44.1 kHz, а stream — к 48 kHz, делить его на 48 000 неправильно. Удобно хранить song time в секундах плюс отображение на выходные frames. Для playback speed, отличной от 1, нужен соответствующий коэффициент/segment mapping.
- **Определить transport-семантику:** момент фактического старта, lead-in, пауза, draining, завершение. До presentation time первого song frame нужен явно заданный preroll, а не предположение, что песня уже звучит. В sounddevice `stop()` ожидает проигрывания pending buffers, `abort()` прекращает обработку без такого ожидания. Ни одна операция не заменяет игровой state machine.
- **При underrun не маскировать проблему сглаживанием:** отметить разрыв, восстановить mapping или перезапустить участок по выбранной политике. Пропуск/вставка тишины меняет отношение между song frames и device time.
- **Не смешивать начала отсчёта.** `perf_counter`, OS input timestamps и PortAudio time — не обязательно одни часы. Для judgment нужна явная привязка временных систем и проверка drift.
- **Разделить аудио-, input- и display-калибровку.** Поправка на монитор не должна незаметно менять время музыкальных событий или дважды применяться к input judgment.

### 4.3. Callback и платформа

- Callback должен целиком заполнять предоставленный выходной буфер, включая нули при отсутствии данных.
- В нём нельзя читать с диска/сети, выполнять модель, строить спектрограммы, печатать логи, ждать mutex или делать непредсказуемо тяжёлые операции. Подготовка PCM и команд — заранее; передача через bounded preallocated buffers/ring.
- **Не вызывать `stream.time` из callback:** использовать переданную структуру `time`. Документация запрещает PortAudio API calls из callback, за исключением разрешённого CPU-load запроса.
- Python callback остаётся soft real-time: GIL, сборщик мусора, другие Python-потоки и планировщик могут нарушить deadline. ML/DSP лучше завершать до игры или выносить в отдельный процесс. Для жёстких гарантий нужен native audio callback, а не другой Python wrapper.
- Музыку и синхронные hitsounds желательно микшировать в одном выходном потоке. Два независимых аудиодвижка создают два transport/latency пути.

Эти требования следуют из callback-контракта sounddevice/PortAudio; архитектурные рекомендации — вывод для игрового приложения. [1](https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html)

**Backend проверяется на целевой ОС:** Windows — сравнить подходящие WASAPI shared/exclusive и ASIO при наличии драйвера/сборки; macOS — CoreAudio; Linux — доступные ALSA/JACK/PipeWire-маршруты. Это план проверки, а не универсальный рейтинг. `WasapiSettings(exclusive=True)` позволяет обойти software mixing, но не гарантирует лучший результат на любом устройстве. Для pip-установки sounddevice на Windows документация отдельно описывает `SD_ENABLE_ASIO`, задаваемую до импорта; ASIO нельзя считать автоматически включённым. [13](https://python-sounddevice.readthedocs.io/en/0.5.6/api/platform-specific-settings.html) [19](https://python-sounddevice.readthedocs.io/en/0.5.6/installation.html)

### 4.4. Минимальный план проверки на железе

1. Зафиксировать ОС, устройство, backend, driver, sample rate, requested и reported latency. Для основного теста использовать проводной выход; Bluetooth измерять отдельно.
2. Начать с автоматического размера callback и low-latency запроса; затем сравнить устойчивые конфигурации. Для ориентира: 128 frames при 48 kHz — 2.67 ms, 256 — 5.33 ms **длительности одного блока**, не полной задержки.
3. Проверять underruns, p50/p95/p99 задержек и ошибку позиции при обычной нагрузке, интенсивном рендере, сборке мусора и фоновой CPU/GPU-нагрузке.
4. Воспроизводить тестовый click track; использовать loopback/внешнюю запись. Loopback обычно измеряет ещё и входной тракт: без отдельной калибровки это не чистая output-only latency.
5. Проверить drift за 15–30 минут, несколько seek/pause/resume, конец трека и смену устройства.
6. Отдельно измерить input и display pipeline. По результатам выбрать **минимальный устойчивый**, а не минимальный запускающийся буфер.

## 5. Генерация карт: onset detection не равен chart generation

Нужны как минимум три уровня:

1. **Музыкальное время:** beats, downbeats, tempo map, локальные onsets и акценты.
2. **Выбор игровых событий:** какие звуки вообще превратить в ноты при данной сложности, где оставить паузы, как развивать плотность.
3. **Lane/action assignment:** куда поставить событие, сделать ли аккорд/hold, как связать его с предыдущими и следующими действиями.

Акустический onset detector ищет начала звуков. Генератор chart onsets выбирает музыкально и игрово уместное подмножество/интерпретацию. Beat tracker даёт доли, но не все ноты мелодии/ударных. Хороший показатель на одной из задач не доказывает качество остальных.

**osu!mania 4K:** дорожки, одновременные нажатия, начала и концы LN, состояние удержания. Это не обычная single-label классификация одного key в каждый момент.

**osu!taiko:** одна лента; важны типы ударов don/katsu, большие ноты, rolls и паттерны. Это не four-lane assignment. Часть решений о том, какой рукой играть обычный удар, возникает у игрока, а не хранится как поле «левая/правая дорожка».

## 6. Релевантные открытые проекты 2023–2026

Лицензия **кода** не означает автоматически ту же лицензию на веса, музыку, обучающие карты и все зависимости. «Код опубликован без найденной лицензии» ниже не называется unrestricted open-source.

| Проект | Реальный целевой режим и результат | Архитектура | Веса / готовность / лицензия |
|---|---|---|---|
| **Mapperatorinator**, 2024–2026 | Все osu! gamemodes, в том числе **mania и taiko**; времена, события, mania columns/LN, timing и прочие атрибуты | Encoder-decoder **Transformer**, модифицированный Whisper; отдельное diffusion-уточнение XY для standard | Инструкции, UI/Colab, pretrained workflow и mode-specific модели. **MIT-код**; условия конкретных весов проверять отдельно. **Первый кандидат на пробу** |
| **MuG Diffusion**, 2023 | **4K VSRG**; тайминг + раскладка, LN, SR/MSD, стиль и паттерны | Audio-conditioned **latent diffusion**, autoencoder + U-Net | Веса предлагаются в Windows bundle, есть запуск из исходников на других платформах. MIT-код, но **противоречивые NC/CC0 условия** для весов/карт. Прежде всего исследовательский/некоммерческий эксперимент |
| **Beat-Aligned Spectrogram-to-Sequence / GOCT**, ISMIR LBD 2023 | **osu!mania 4K**, timing + action tokens; также timing-эксперименты StepMania | Beat-aligned encoder-decoder **Transformer** | MIT, ссылка на pretrained archive около 300 MB. Исследовательские скрипты, старое окружение; не turnkey SDK |
| **AutoOsu**, ISMIR LBD 2023 | **mania 4K**, совместная генерация placements и действий/колонок | CNN + **BiGRU/GRU**, две совместно обученные ветви, обе получают audio features | Несколько `.pt` checkpoint присутствуют в репозитории. **Лицензия не найдена**; не переносить в продукт без разрешения |
| **GenéLive!**, AAAI 2023 | Открытая воспроизводимая часть — **onset generation** на StepMania, вывод в MIDI | Multi-scale CNN + **BiLSTM**, beat guide | Код MIT, checkpoint в репозитории. Полезный timing baseline, **не готовая открытая модель mania lanes** |
| **ITGPT**, 2026 | **DDR/ITG 4-panel**, step placement + step selection | Иерархический **Transformer** для onset и autoregressive Transformer для действий | MIT, training/generation code, HF demo; README направляет к GUI для генерации. Не проверялся локальный комплект pretrained-весов. Перенос на клавиатурную mania требует проверки домена |
| **Keyloom**, 2026 | **mania 4K tap-only**, audio features → планирование секций/фраз → count/key placement → `.osz` | Составной learned/rule-based pipeline с MusicFM-признаками и отдельным планированием | MIT-код, **реальный model bundle в GitHub Release v0.2.0**. Только постоянный BPM, **нет LN и полноценной variable-tempo поддержки**; full training pipeline/корпус не опубликованы |
| **rhythm-chart-diffusion**, POSTECH UGRP 2026 | **mania 4K**, дискретные состояния нот, audio/difficulty/style conditioning | **Masked discrete diffusion**, cross-attention denoiser | MIT, pre-alpha. Модель и sampler **реализованы**, авторы описывают обучение и validation; это уже не одни stubs. Но публичная загрузка checkpoint не обнаружена; AR baseline отложен, исследование не завершено |

Первичные репозитории и инструкции: [5](https://github.com/OliBomby/Mapperatorinator) [6](https://github.com/Keytoyze/Mug-Diffusion) [7](https://github.com/issyun/AutoOsu) [8](https://github.com/stet-stet/goct_ismir2023) [9](https://github.com/KLab/AAAI-23.6040) [10](https://github.com/miguelomalley/ITGPT) [20](https://github.com/ZhSM27/Keyloom) [21](https://github.com/qxxiit/rhythm-chart-diffusion).

### 6.1. Mapperatorinator: наиболее практичный первый эксперимент

- В README прямо заявлены все gamemodes, а postprocessing включает **перевод mania column events в X** и построение taiko drumrolls. Это не вывод из слова «osu!» в названии.
- В актуальном описании — Transformer около **219M параметров**, mel spectrogram input, разреженный event vocabulary, conditioning по gamemode, сложности, mapper/style и metadata.
- Длинная песня обрабатывается перекрывающимися окнами; есть resnap времён к музыкальной сетке. Это не означает гарантированно безошибочный BPM/offset.
- Для mania доступны настройки keycount и hold-note ratio; для taiko — отдельный gamemode. Release v32 описывает mode-specific fine-tunes и автоматический выбор модели.
- Diffusion refinement в архитектуре предназначен прежде всего для **XY-координат standard**. Его наличие нельзя трактовать как доказательство превосходства diffusion в mania lane assignment.
- «Rankable quality» — цель проекта, не независимый benchmark. Скорость, VRAM, корректность выбранных весов и качество на своей музыке нужно проверить отдельно. [5](https://github.com/OliBomby/Mapperatorinator) [22](https://github.com/OliBomby/Mapperatorinator/releases/tag/v32.0.0)

### 6.2. MuG Diffusion: очень релевантен 4K, но есть лицензионный стоп-фактор

README заявляет 4K VSRG, управление SR/MSD, LN ratio и семействами паттернов Etterna. Конфигурация модели подтверждает autoencoder/U-Net/audio-conditioning pipeline, а не diffusion только пространственных координат. Веса предлагается извлечь из bundled executable. [6](https://github.com/Keytoyze/Mug-Diffusion) [23](https://github.com/Keytoyze/Mug-Diffusion/blob/c2903d4/configs/mug/mug_diffusion.yaml)

В разделе Credits одновременно написано:

> Charts created through MuG Diffusion ... CC0 1.0 ...
>
> The model weights and charts created are non-commercial.

CC0 и такое non-commercial ограничение для одних и тех же карт противоречат друг другу. **Не делать вывод «можно свободно в коммерческую игру» по MIT-бейджу кода; запросить ясные условия у автора.** Саму доступность внешнего bundle и его работу здесь не проверяли. [6](https://github.com/Keytoyze/Mug-Diffusion)

### 6.3. AutoOsu, GOCT и GenéLive!: разные baseline, не взаимозаменяемые

**AutoOsu.** Официальная программа ISMIR описывает совместно обучаемые recurrent layers для onset detection и action generation, обе получают аудиопризнаки. Исходник использует GRU, а не LSTM. Inference принимает заданные BPM, offset и difficulty; beat embedding рассчитан на четыре доли. Это полезный компактный baseline полной 4K-задачи, но опубликованный код без найденной лицензии нельзя считать разрешением на произвольное переиспользование. [7](https://github.com/issyun/AutoOsu) [24](https://ismir2023program.ismir.net/lbd_319.html)

**GOCT / Beat-Aligned.** Работа использует beat-normalized log-mel input и time/action tokens для 4K, включая удержания. На osu!mania в статье получено timing F1 **84.6%** против **83.9%** C-LSTM. **Это сравнение по временным событиям: сравниваемые модели обучены без action tokens. Число не доказывает точность выбора дорожек.** На маленьких Fraxtil/ITG без pretraining Transformer уступает baseline; transfer улучшает результат. Сами авторы отмечают проблемы очень разреженных лёгких карт. [8](https://github.com/stet-stet/goct_ismir2023) [25](https://arxiv.org/html/2311.13687)

**GenéLive!** Сильная сторона — beat guide, multi-scale temporal context и проверенный авторами production-assisted workflow. Но статья явно сосредоточена на **onset model**, а открытый repository воспроизводит StepMania-эксперименты и выдаёт MIDI. Закрытые Love Live! данные и полноценный action pipeline не следует считать частью открытого lane generator. Препринт появился в 2022, конференционная публикация — **AAAI 2023**. [9](https://github.com/KLab/AAAI-23.6040) [26](https://arxiv.org/html/2202.12823v2)

### 6.4. Новые 2026-проекты: что уже есть и чего нет

**Keyloom:** Release v0.2.0 от 4 ноября 2026 содержит downloadable `keyloom-models-r2.zip` около **166 MiB**. Это полезный современный 4K-workbench для tap-карт, но не замена full mania-модели с LN. Model cards прямо отделяют воспроизводимость inference от воспроизводимости training и не заявляют независимую сертификацию музыкального качества. [20](https://github.com/ZhSM27/Keyloom) [27](https://github.com/ZhSM27/Keyloom/releases/tag/v0.2.0)

**qxxiit/rhythm-chart-diffusion:** актуальные README и `src/models/diffusion.py` содержат настоящую denoiser-реализацию, sampler и generate/evaluate workflow. В экспериментальном журнале есть full-data training и audio ablations. При этом статус остаётся pre-alpha, проект планируется до января 2027, AR baseline отложен. На дату проверки GitHub Releases пуст и ссылка на публичный trained checkpoint в изученном README не найдена. Поэтому это **интересная исследовательская кодовая база**, но не подтверждённый готовый победитель сравнения Transformer vs Diffusion. Training/validation CE и авторские эвристики не заменяют независимую игровую оценку. [21](https://github.com/qxxiit/rhythm-chart-diffusion) [28](https://github.com/qxxiit/rhythm-chart-diffusion/blob/10805a6/docs/EXPERIMENTS.md)

**ITGPT:** препринт июля 2026 и MIT-репозиторий предлагают генерацию timing и step selection для DDR/ITG. Для дорожечной игры это более близкий источник, чем osu!standard XY generation. Но ограничения ног/танцевальной площадки и клавиатурной mania различаются; reported DDR/ITG результаты не являются mania benchmark. [10](https://github.com/miguelomalley/ITGPT) [29](https://arxiv.org/html/2607.14148v1)

### 6.5. Другой дорожечный домен: STRUM, 2026

[STRUM](https://github.com/opria123/strum) генерирует Clone Hero / YARG charts: drums, guitar/bass, vocals и keys. Использует source separation, CRNN onset detection, классификацию ударных, а для guitar/bass — в том числе pitch tracking и **правила** отображения в frets. Поэтому «AI lane assignment» здесь не означает, что весь mapping обучен end-to-end.

В статье заявлены code/weights и benchmark; README также заявляет MIT, но отдельная распознанная GitHub лицензия отсутствовала, GitHub Releases на дату проверки пуст. Полноту внешней раздачи весов не подтверждали. Главное ограничение сравнения: onset F1 оценивается с **±100 ms и per-song global-offset search**, на предварительно отобранном operating-envelope наборе; это не сопоставимо напрямую с 10/25 ms mania-тестом и не проверка абсолютного playback offset. В статье указан 30-song benchmark, README отчёт — n=29; прежде чем повторять цифры, нужно зафиксировать ревизию и manifest. [30](https://arxiv.org/html/2605.12135v1) [31](https://github.com/opria123/strum)

### 6.6. Что не стоит ошибочно рекомендовать как готовый mania/Taiko generator

- **osu!dreamer** — diffusion-проект **osu!standard**; может быть интересен архитектурно, но это не проверенная модель колонок mania. [32](https://github.com/jaswon/osu-dreamer)
- **OliBomby/osu-diffusion** — прежде всего генерация/уточнение **XY** с заданными rhythm/spacing, не полный audio-to-mania pipeline. [33](https://github.com/OliBomby/osu-diffusion)
- **BeatLearning** — Transformer WIP; в README mania/Taiko остаются в roadmap, а рабочий pipeline ограничен standard/simple hit objects. Сам README ссылается на Mapperatorinator как более новый проект. [34](https://github.com/sedthh/BeatLearning)
- **ManiaMapper** — релевантная 4K CNN/BiLSTM/LSTM-база, но README говорит об отсутствии pretrained weights и небольшом собственном training set. Training accuracy не доказывает качество карт на новых песнях. [35](https://github.com/rapidvind/ManiaMapper)
- **TaikoNation (2021)** и **DDC (2017)** полезны как исторические baseline, но не являются сами по себе ответом на поиск новых работ 2023–2026.

## 7. Важные статьи 2025–2026 и ограничения их выводов

### Time-based Chart Partitioning — AIIDE 2025

Комбинирует AutoOsu onset head, **beat snapping** и подбор фрагментов human charts для note-type/lane patterns. Работа показывает, что гибрид «нейросеть + музыкальная сетка + pattern retrieval» может быть полезнее, чем попытка получить всё сырым нейросетевым выходом. [36](https://ojs.aaai.org/index.php/AIIDE/article/view/36808)

При этом:

- improved onset precision и local pattern recall не означают доказанное превосходство по всем свойствам раскладки;
- onset+note-type F1 у TCP и AutoOsu в наиболее permissive сравнении статистически не различался;
- окна допуска и beat-snapped evaluation требуют внимательного чтения;
- human playtest в работе отсутствует;
- отдельная опубликованная лицензированная реализация TCP в ходе проверки не найдена. Это статья для проектирования hybrid baseline, а не уверенно готовая OSS-зависимость. [37](https://ojs.aaai.org/index.php/AIIDE/article/download/36808/38946)

### Transfer Learning for Rhythm Game Chart Generation — октябрь 2026

Замораживает pretrained audio encoder, в том числе **MuScriptor**, и обучает projection + Transformer decoder. Исследует standard, taiko и 4K mania. На standard event F1 с MuScriptor против encoder той же архитектуры с нуля растёт **0.323 → 0.433**. В mania joint training с другими osu! режимами даёт **0.492 → 0.506** event F1; добавление Clone Hero вместо этого снижает показатель до **0.482**. [38](https://arxiv.org/html/2610.14955)

Но **event F1 в этой статье проверяет время и тип события, не колонку mania**; это прямо оговорено в разделе Evaluation. При joint training также ухудшаются показатели hold-start/hold-end. Следовательно, это аргумент изучать transfer/audio frontend, а не доказательство решённой задачи lane assignment или универсальной пользы любых дорожечных данных. Публичный готовый chart checkpoint и однозначная ссылка на chart-generation код в изученной статье/поиске не обнаружены; не включаю работу в список проверенно доступных моделей. [38](https://arxiv.org/html/2610.14955)

### ChartGenEval — июль 2026

Полезен прежде всего для **оценки**, а не как generator. Работа показывает, почему agreement с одной картой, perplexity и self-similarity не образуют универсальный quality score; предлагает раздельные сигналы и проверки искусственными повреждениями. Применимость к своей игре всё равно нужно валидировать. [39](https://arxiv.org/html/2607.12857v1)

## 8. CRNN, Transformer или Diffusion?

Это оценка для выбора инженерного направления, а не общий SOTA-рейтинг.

| Семейство | Onset / timing | Lane / action assignment | Основные компромиссы | Когда выбрать |
|---|---|---|---|---|
| **CRNN: CNN + GRU/LSTM** | Сильный понятный baseline; audio features + beat phase + difficulty; возможна framewise onset probability | Отдельная/совместная recurrent ветвь с audio и action history, либо constrained pattern stage | Ограниченный контекст без специальных приёмов; autoregressive ошибки; BiRNN не causal без lookahead | Небольшой исследовательский baseline, CPU-oriented inference после замеров, прозрачная декомпозиция timing/patterns |
| **Transformer** | Event tokens позволяют не классифицировать огромное количество пустых frames; beat alignment и хороший frontend имеют большое значение | Естественно кодирует time, chord/column/action, hold start/end, style; позволяет conditioning и grammar masks | Данные и compute; длина контекста/окна; AR decode; invalid tokens без ограничений | **Предпочтительный первый путь для полной условной карты**, особенно с готовым релевантным checkpoint |
| **Diffusion / discrete diffusion** | Совместно уточняет представление событий/сетку; полезен для offline conditional generation | Может моделировать многоклавишный tensor/паттерн целиком, поддерживать разнообразие и conditioning | Несколько denoising steps; legal-state decoding; временная дискретизация; наличие весов и качество сильно зависят от проекта | 4K offline-эксперименты, несколько вариантов карт, style/pattern control; не обязательный выбор для первого MVP |

Основания: AutoOsu, GenéLive!, Beat-Aligned, ITGPT, MuG и реализация discrete diffusion. **Diffusion — это процедура генерации/обучения, Transformer — архитектура:** denoiser может сам быть Transformer. Поэтому эти категории не всегда взаимоисключающие. [6](https://github.com/Keytoyze/Mug-Diffusion) [7](https://github.com/issyun/AutoOsu) [9](https://github.com/KLab/AAAI-23.6040) [21](https://github.com/qxxiit/rhythm-chart-diffusion) [25](https://arxiv.org/html/2311.13687) [29](https://arxiv.org/html/2607.14148v1)

Не найден честный единый benchmark всех трёх семейств на одинаковых песнях, сложности, timing tolerance, колонках, LN и human playtests, позволяющий назвать универсального победителя. Это ограничение проведённого поиска, а не утверждение, что такой эксперимент нигде не существует.

## 9. torchaudio, librosa и madmom

| Инструмент | Что реально даёт для onsets | Что даёт для дорожек | Практическая оценка |
|---|---|---|---|
| **librosa** | DSP-признаки, spectral-flux onset strength, peak picking, beat/tempo, удобная offline-диагностика | Нет готового learned lane generator | Лучший простой **контрольный DSP baseline** и инструмент проверки временных преобразований; не нейросетевой SOTA сам по себе |
| **torchaudio** | Tensor transforms, mel/STFT, resampling, augmentation; компоненты для собственного PyTorch frontend | Можно построить модель поверх features, но стандартного mania/Taiko lane mapper нет | Удобен рядом с PyTorch, в том числе для batched/device processing; учитывать текущий maintenance status и переход I/O в TorchCodec |
| **madmom** | Готовые CNN/RNN onset processors, beat/downbeat models и decoding | Готового lane assignment нет | Полезен как pretrained MIR baseline, но старый packaging, совместимость и **NC-лицензии весов** осложняют продукт |

### librosa

`onset_detect()` ищет пики onset envelope; `onset_strength()` считает spectral flux. Это не обученный chart model. В source `backtrack=True` описан прежде всего как способ найти точки нарезки аудио: он не является универсальным улучшением judgment timing. [40](https://github.com/librosa/librosa/blob/76f10b8/librosa/onset.py)

Для ритм-игры важно явно фиксировать `sr`, `hop_length`, mono/stereo преобразование, padding/centering и единицы времени. При 22 050 Hz и hop 512 шаг сетки около **23.22 ms** — уменьшение hop может улучшить дискретизацию, но не гарантирует точность детектора. Сам `librosa.load()` по умолчанию может изменить sample rate: нельзя сравнивать индекс такого массива с playback frames другого формата. [41](https://github.com/librosa/librosa/blob/76f10b8/librosa/core/audio.py)

На дату проверки PyPI: **librosa 1.0.1**, опубликована 12 ноября 2026. Это очень свежий релиз; старые research repos не следует без проверки обновлять на latest. [42](https://pypi.org/pypi/librosa/json)

### torchaudio

Официальная документация сообщает о переходе в **maintenance phase** с 2.8 и удалении части API в 2.9. Начиная с 2.9, `torchaudio.load()` использует **TorchCodec AudioDecoder**, а некоторые старые параметры, включая `normalize`, `buffer_size`, `backend`, сохраняются только для совместимости и игнорируются. Для нового кода разработчики рекомендуют прямой TorchCodec API. Это не означает, что mel/resample transforms перестали быть полезны. [43](https://docs.pytorch.org/audio/main/index.html) [44](https://docs.pytorch.org/audio/stable/generated/torchaudio.load.html)

На дату проверки PyPI: **torchaudio 2.11.1**, 28 октября 2026. Среду исследования нужно фиксировать согласованно с PyTorch/TorchCodec и инструкциями выбранной модели; «установить всё самое новое» не равно воспроизводимости. [45](https://pypi.org/pypi/torchaudio/json)

### madmom

Последний PyPI-релиз — **0.16.1 от 14 ноября 2018**, а не новая модель 2023–2026. Наличие нейросетевых onset/beat processors не делает его свежим универсальным SOTA. Совместимость с современными Python/NumPy нужно проверять; разработчики Beat This! отдельно предупреждают о старых ограничениях PyPI-версии madmom. [46](https://pypi.org/pypi/madmom/json) [47](https://github.com/CPJKU/beat_this)

Особенно важно разделение лицензий:

- код — BSD;
- shipped data/model files — **CC BY-NC-SA 4.0**;
- LICENSE отдельно требует обращаться за разрешением, если технология с этими моделями включается в коммерческий продукт. [48](https://github.com/CPJKU/madmom/blob/27f032e/LICENSE)

### Дополнение: Beat This!, ISMIR 2024

Для нового beat/downbeat frontend стоит проверить **Beat This!**, а не ограничиваться старым madmom. Есть pretrained-модели, CPU/GPU inference и официальный MIT для кода **и опубликованных весов**, с оговоркой авторов об исходных training data. Использование `dbn=False` не требует madmom DBN. Но это **beat/downbeat tracking, не all-onset detection и не lane assignment**. [47](https://github.com/CPJKU/beat_this)

## 10. Предлагаемый стек и порядок работ

### 10.1. Если цель — быстро проверить игровую идею

```text
Музыкальный файл
  └─ Decode → canonical PCM + явный track offset
       ├─ Offline features / beat grid / генератор карты
       │    └─ Проверка тайминга, сложности и legal actions
       │         └─ Кэш: time, lane/action, duration, confidence/metadata
       └─ Один audio output stream: музыка + hitsounds
            └─ DAC-anchored song clock
                 ├─ Rendering
                 └─ Input judgment с явным clock mapping/калибровкой
```

**Мой первый набор:**

- pygame либо другой UI/rendering framework — только окно, графика и input;
- **sounddevice `OutputStream` / `RawOutputStream`** — PCM output;
- отдельный offline decoder и resampling, одинаковая временная база для анализа и playback;
- **Mapperatorinator** — первая полная карта, с явным выбором mania/taiko checkpoint;
- **librosa** — диагностический DSP baseline и визуализация;
- при необходимости независимого beat grid — **Beat This!**;
- generation/ML — до начала gameplay или отдельным процессом, не в audio callback.

Для 4K diffusion A/B-теста добавить MuG только после прояснения лицензии. Для permissive research-кода discrete diffusion изучить qxxiit, понимая отсутствие найденного публичного checkpoint. Keyloom полезен, если MVP сознательно ограничен constant-tempo tap-only 4K.

### 10.2. Если нужна своя контролируемая модель

1. Начать с явного представления: `(time, lane/action, duration)` и состояние удержаний.
2. Сравнить два честных baseline: **DSP onset + constrained pattern selector** и **CRNN onset + audio-aware lane/action head**.
3. Для 4K не сводить выход к `argmax` одной колонки: нужны simultaneous notes и LN-state. Подходят row/action tokens либо multi-label/stateful heads с запретом невозможных переходов.
4. Для taiko использовать отдельный vocabulary/constraints; не выдавать стандартный XY decoder или piano-pitch transcription за taiko pattern model.
5. Сложность должна влиять не только на порог onset detection: нужны паузы, плотность по секциям, размер аккордов, hand strain и допустимые паттерны.
6. После воспроизводимого baseline пробовать Transformer и затем diffusion. Не тратить первый этап на большой training run до проверки датасета, timing offsets и gameplay constraints.
7. Для игры наподобие Melody’s Escape дополнительно определить словарь собственных препятствий/действий и физические ограничения персонажа. Колонка osu!mania сама по себе ещё не означает допустимый прыжок, перекат или переход между дорожками.

Последний пункт — архитектурное предложение для собственного проекта, не утверждение об устройстве генератора Melody’s Escape 2.

### 10.3. Как сравнивать модели, не обманывая себя

- Делить train/validation/test **по песне**, объединяя её версии, cut/full-size, разные сложности и совпадающие аудиофайлы. Дополнительно полезен artist-holdout.
- Отдельно считать onset precision/recall/F1, например при 10/25/50 ms, и ошибку абсолютного offset. Это предлагаемый протокол, не смешение чужих таблиц.
- Отдельно считать **lane-aware event F1**, chord matching, hold-start/end и duration errors. Явно написать, учитывает ли метрика колонку.
- Проверять invalid holds, пересечения, запрещённые комбинации, скорость повторов, hand strain, целевую сложность и распределение плотности по секциям.
- Оценивать повторы/фразы, разнообразие и соответствие музыкальным акцентам, но не оптимизировать одну perplexity/self-similarity как «качество».
- Сравнивать runtime, RAM/VRAM и число вариантов на одном hardware; не переносить авторский benchmark на своё устройство.
- Добавить blind playtest с несколькими игроками/мапперами. Одна human chart — не единственная правильная раскладка; зеркальная или стилистически другая карта может быть хорошей.

## 11. Зафиксированные ревизии ключевых репозиториев

Даты ниже — дата последнего default-branch commit, полученного через GitHub API с ограничением до конца **2026-11-15 UTC**, не `updated_at`, не показатель качества и не обещание поддержки. Короткие SHA в ссылках позволяют восстановить исследованное состояние.

| Репозиторий | Commit date UTC | Ревизия |
|---|---|---|
| Mapperatorinator | 2026-11-13 | [d31a634](https://github.com/OliBomby/Mapperatorinator/tree/d31a634) |
| MuG Diffusion | 2023-06-05 | [c2903d4](https://github.com/Keytoyze/Mug-Diffusion/tree/c2903d4) |
| AutoOsu | 2023-09-29 | [b81dc6f](https://github.com/issyun/AutoOsu/tree/b81dc6f) |
| Beat-Aligned / GOCT | 2024-01-29 | [c8df09b](https://github.com/stet-stet/goct_ismir2023/tree/c8df09b) |
| GenéLive! | 2023-04-26 | [43806e5](https://github.com/KLab/AAAI-23.6040/tree/43806e5) |
| ITGPT | 2026-07-17 | [81de0bf](https://github.com/miguelomalley/ITGPT/tree/81de0bf) |
| Keyloom | 2026-11-04 | [cdbea0f](https://github.com/ZhSM27/Keyloom/tree/cdbea0f) |
| rhythm-chart-diffusion | 2026-10-15 | [10805a6](https://github.com/qxxiit/rhythm-chart-diffusion/tree/10805a6) |
| STRUM | 2026-05-10 | [9f420cb](https://github.com/opria123/strum/tree/9f420cb) |
| pyminiaudio | 2026-11-07 | [e84ad4c](https://github.com/irmen/pyminiaudio/tree/e84ad4c) |
| Beat This! | 2026-05-28 | [b95c8ab](https://github.com/CPJKU/beat_this/tree/b95c8ab) |

**Оставшиеся вопросы перед реализацией:** целевая ОС/устройство, допустимая задержка hitsounds, нужные типы действий и holds, offline или live generation, доступная GPU/CPU и коммерческое использование. Они не мешают выбрать исследовательское направление, но необходимы перед обещанием численной latency и окончательным выбором checkpoint.
