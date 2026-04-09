#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from nlp_analysis import analyze_text

# Test the function
test_texts = [
    "I just achieved my goal of running a marathon! It was an amazing success and I'm so proud of myself.",
    "This product is terrible. It broke after one day and customer service was unhelpful.",
    "The new AI technology is revolutionary. It will change how we work forever.",
    "Today was okay, nothing special happened."
]

print("Testing NLP Analysis Function:")
print("=" * 50)

for i, text in enumerate(test_texts, 1):
    print(f"\nTest {i}:")
    print(f"Input: {text}")
    result = analyze_text(text)
    print(f"Output: {result}")
    print("-" * 30)