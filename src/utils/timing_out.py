

def timing_out(timing_report):
    print("\n" + "=" * 40)
    print("=" + " "*12 +"TIMING REPORT" + " "*13 +"=")
    print("=" * 40)
    for step, duration in timing_report.items():
        print(f"{step}: {duration:.2f} sec")
    print("=" * 40)