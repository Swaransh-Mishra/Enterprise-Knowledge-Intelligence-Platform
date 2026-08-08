from app.text_cleaner import TextCleaner

cleaner = TextCleaner()

sample = """
Hello



This      is      a     test.


Another        line.






End.
"""

print("=" * 50)
print("RAW TEXT")
print("=" * 50)

print(sample)

print()

print("=" * 50)
print("CLEAN TEXT")
print("=" * 50)

print(cleaner.clean(sample))