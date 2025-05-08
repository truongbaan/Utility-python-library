#this is just about langdetect library, since they already simple enough, there is no need to make function 

from langdetect import detect, detect_langs #need pip install langdetect

if __name__ == "__main__":
    text1 = "This is a sample text in English. Xin chào bạn đẹp trai "
    detected_language1 = detect(text1)
    print(f"The language of '{text1}' is: {detected_language1}")

    text2 = "Ceci est un exemple de texte en français."
    detected_language2 = detect_langs(text2) #it return a list
    print(f"The language of '{text2}' is: {detected_language2}")

    text3 = "Đây là một đoạn văn bản mẫu bằng tiếng Việt."
    detected_language3 = detect_langs(text3)

    if detected_language3:
        most_likely_language = detected_language3[0]
        language_code = most_likely_language.lang
        probability = most_likely_language.prob

        print(f"The detected language of '{text3}' is: {language_code}")
        print(f"The confidence score is: {probability:.4f}")

        extracted_values = [language_code, probability]
        print(f"Extracted values as a list: {extracted_values}")
    else:
        print(f"Could not detect the language of '{text3}'.")