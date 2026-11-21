with open(r'C:\Users\huynh\.gemini\antigravity-ide\brain\fab9a45a-3280-4c2a-877d-f01a97105f33\.system_generated\steps\53\content.md', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('getPageFoodData')
print(text[pos-800:pos+800])
