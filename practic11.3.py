import nltk
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string
import matplotlib.pyplot as plt
print("Завантаження ресурсів NLTK...")
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  
print("Ресурси завантажені успішно!\n")
text = gutenberg.raw('melville-moby_dick.txt')
words = word_tokenize(text)
total_words = len(words)
print(f"Загальна кількість слів у тексті: {total_words}")
fdist = FreqDist(words)
top_10_words = fdist.most_common(10)
print("\n10 найбільш вживаних слів у тексті:")
for word, count in top_10_words:
    print(f"'{word}': {count}")
plt.figure(figsize=(12, 6))
words_list, counts = zip(*top_10_words)
plt.bar(words_list, counts, color='skyblue')
plt.title('10 найбільш вживаних слів у тексті Moby Dick')
plt.xlabel('Слова')
plt.ylabel('Частота')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
filtered_words = []
for word in words:
    word_lower = word.lower()
    if (word_lower not in stop_words and
        word not in punctuation and
        word.isalpha()):
        filtered_words.append(word_lower)
filtered_total_words = len(filtered_words)
print(f"\nКількість слів після видалення стоп-слів та пунктуації: {filtered_total_words}")
fdist_filtered = FreqDist(filtered_words)
top_10_filtered_words = fdist_filtered.most_common(10)
print("\n10 найбільш вживаних слів після видалення стоп-слів та пунктуації:")
for word, count in top_10_filtered_words:
    print(f"'{word}': {count}")
plt.figure(figsize=(12, 6))
words_list_filtered, counts_filtered = zip(*top_10_filtered_words)
plt.bar(words_list_filtered, counts_filtered, color='lightcoral')
plt.title('10 найбільш вживаних слів у тексті Moby Dick (після фільтрації)')
plt.xlabel('Слова')
plt.ylabel('Частота')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print(f"\nДодаткова інформація:")
print(f"Відсоток слів, що залишилися після фільтрації: {filtered_total_words/total_words*100:.2f}%")
