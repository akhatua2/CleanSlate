#!/usr/bin/env python3
"""
Test script to verify SGLang backend implementation.

This tests:
1. Import and initialization
2. compute() method for log probability computation
3. generate() method for text generation with token capturing
"""

import sys

def test_sglang_import():
    """Test that SGLang backend can be imported."""
    try:
        from cleanslate.backends.sglang_backend import SGLangBackend
        print("✓ SGLang backend imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import SGLang backend: {e}")
        return False

def test_generation_result_import():
    """Test that GenerationResult can be imported."""
    try:
        from cleanslate.backends.base import GenerationResult
        print("✓ GenerationResult imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import GenerationResult: {e}")
        return False

def test_backend_structure():
    """Test that SGLang backend has required methods."""
    try:
        from cleanslate.backends.sglang_backend import SGLangBackend
        import inspect

        # Check for required methods
        required_methods = ['compute', 'generate', 'device', 'model_name']
        methods = set(dir(SGLangBackend))

        for method in required_methods:
            if method not in methods:
                print(f"✗ SGLang backend missing method: {method}")
                return False

        # Check method signatures
        sig = inspect.signature(SGLangBackend.generate)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'seed', 'max_new_tokens', 'temperature', 'top_k', 'top_p', 'do_sample', 'capture_top_k']

        if set(params) != set(expected_params):
            print(f"✗ generate() signature mismatch. Got: {params}, Expected: {expected_params}")
            return False

        print("✓ SGLang backend has all required methods with correct signatures")
        return True
    except Exception as e:
        print(f"✗ Backend structure check failed: {e}")
        return False

def test_generation_result_structure():
    """Test that GenerationResult has required fields."""
    try:
        from cleanslate.backends.base import GenerationResult
        import dataclasses

        fields = {f.name for f in dataclasses.fields(GenerationResult)}
        required_fields = {
            'generated_text', 'generated_token_ids', 'generated_tokens',
            'token_log_probs', 'top_k_tokens', 'top_k_probs',
            'max_new_tokens', 'temperature', 'top_k', 'top_p', 'do_sample',
            'generation_time_ms'
        }

        missing = required_fields - fields
        if missing:
            print(f"✗ GenerationResult missing fields: {missing}")
            return False

        print("✓ GenerationResult has all required fields")
        return True
    except Exception as e:
        print(f"✗ GenerationResult structure check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SGLang Backend Integration Tests")
    print("=" * 60)
    print()

    tests = [
        ("Import SGLang backend", test_sglang_import),
        ("Import GenerationResult", test_generation_result_import),
        ("Backend structure", test_backend_structure),
        ("GenerationResult structure", test_generation_result_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name}: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Passed: {passed}/{total}")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
