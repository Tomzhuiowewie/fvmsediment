from src.train import run_hump_evolution_test

if __name__ == '__main__':
    import sys
    regime = sys.argv[1] if len(sys.argv) > 1 else 'slow'
    run_hump_evolution_test(regime=regime)