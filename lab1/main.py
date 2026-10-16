import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np


def read_image(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path, image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Не удалось сохранить изображение: {path}")
    Path(path).write_bytes(data.tobytes())


def chroma_key_opencv(foreground, background, lower, upper):
    mask = cv2.inRange(foreground, np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))
    result = foreground.copy()
    cv2.copyTo(background, mask, result)
    return result, mask


def chroma_key_python(foreground, background, lower, upper):
    height, width = foreground.shape[:2]
    result = foreground.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            red, green, blue = foreground[y, x]
            if (lower[0] <= red <= upper[0]
                    and lower[1] <= green <= upper[1]
                    and lower[2] <= blue <= upper[2]):
                result[y, x] = background[y, x]
                mask[y, x] = 255
    return result, mask


def benchmark(algorithm, foreground, background, lower, upper, repeats):
    algorithm(foreground, background, lower, upper)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        result, mask = algorithm(foreground, background, lower, upper)
        times.append((perf_counter() - start) * 1000)
    return result, mask, times


def main():
    parser = argparse.ArgumentParser(description="Замена фона по диапазону RGB")
    parser.add_argument("foreground", type=Path, help="Изображение с однотонным фоном")
    parser.add_argument("background", type=Path, help="Новый фон")
    parser.add_argument("--lower", type=int, nargs=3, default=[0, 100, 0],
                        metavar=("R", "G", "B"), help="Нижняя граница RGB")
    parser.add_argument("--upper", type=int, nargs=3, default=[120, 255, 120],
                        metavar=("R", "G", "B"), help="Верхняя граница RGB")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="Число замеров после одного прогревочного запуска")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("Число замеров должно быть положительным")

    if any(not 0 <= low <= high <= 255
           for low, high in zip(args.lower, args.upper)):
        parser.error("Границы каждого канала должны удовлетворять 0 <= lower <= upper <= 255")

    try:
        foreground = read_image(args.foreground)
        background = read_image(args.background)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Ошибка: {error}\n")

    height, width = foreground.shape[:2]
    background = cv2.resize(background, (width, height))
    result_opencv, mask_opencv, times_opencv = benchmark(
        chroma_key_opencv, foreground, background,
        args.lower, args.upper, args.repeats)
    result_python, mask_python, times_python = benchmark(
        chroma_key_python, foreground, background,
        args.lower, args.upper, args.repeats)

    if not (np.array_equal(result_opencv, result_python)
            and np.array_equal(mask_opencv, mask_python)):
        raise RuntimeError("Результаты реализаций не совпадают")

    args.output.mkdir(parents=True, exist_ok=True)
    save_image(args.output / "result_opencv.png", result_opencv)
    save_image(args.output / "result_python.png", result_python)
    save_image(args.output / "mask.png", mask_opencv)
    preview = np.hstack((foreground, background, result_opencv))
    save_image(args.output / "comparison.png", preview)

    with (args.output / "timings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "opencv_ms", "python_ms"])
        for run, (opencv_ms, python_ms) in enumerate(
                zip(times_opencv, times_python), start=1):
            writer.writerow([run, f"{opencv_ms:.6f}", f"{python_ms:.6f}"])

    opencv_median = median(times_opencv)
    python_median = median(times_python)
    print(f"Размер: {width} x {height}. Замеров: {args.repeats}")
    print("Время алгоритмов без чтения, изменения размера и сохранения файлов:")
    print(f"{'Реализация':<14} {'Минимум, мс':>14} {'Медиана, мс':>14}")
    print(f"{'OpenCV':<14} {min(times_opencv):>14.3f} {opencv_median:>14.3f}")
    print(f"{'Python':<14} {min(times_python):>14.3f} {python_median:>14.3f}")
    print(f"Ускорение OpenCV по медианам: {python_median / opencv_median:.1f} раз")
    print(f"Результаты совпадают. Изображения сохранены в {args.output.resolve()}")


if __name__ == "__main__":
    main()
