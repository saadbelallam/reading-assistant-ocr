

def timing_out(timing_report):
    print("\n" + "=" * 40)
    print("=" + " "*12 +"TIMING REPORT" + " "*13 +"=")
    print("=" * 40)
    total_width = 40

    for step, duration in timing_report.items():
        spacing = total_width - len(step) - len(f"{duration:.2f} s")  # Compute exact spacing
        print(f"{step}{' ' * spacing}{duration:.2f} s")
        
    print("=" * 40)