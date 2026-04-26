from src.config import load_config
from src.runtime import RecognitionRuntime


def print_startup_metrics(runtime: RecognitionRuntime) -> None:
    snapshot = runtime.snapshot()
    metrics = snapshot.metrics

    print("\nRuntime metrics")
    print(f"- registered people: {metrics.registered_people}")
    print(f"- video source: {metrics.video_source}")
    print(f"- model runtime: {metrics.runtime_mode}")
    print(
        "- recognition window: "
        f"{runtime.config['recognition']['recognition_window_frames']} frames"
    )
    print("- biometric access control: disabled; this MVP only identifies and logs events")


def main() -> None:
    config = load_config()
    runtime = RecognitionRuntime(config=config, display_window=True)

    try:
        print("\nStarting biometric access control MVP...")
        print("Press Q in the video window to exit.\n")
        runtime.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print_startup_metrics(runtime)
        print("Resources released. Window closed.")


if __name__ == "__main__":
    main()
