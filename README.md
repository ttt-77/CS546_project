###Data Preprocess
Because we need to use Stanford corenlp toolkit, which is time-consuming, you can download it from this link: https://drive.google.com/drive/folders/1pFUvTbykdTQW1WOjBOSiZyyuwQwcFjh0?usp=sharing

Otherwise, 
1. You need to download dataset from https://github.com/microsoft/CodeT/tree/main/DIVERSE/data/gsm8k. 
2. Then run python preprocess.py on downloaded dataset.
3. Download Stanford corenlp toolkit and open the server.
4. Run python process.py on dataset generated from step2.
5. Run python pre_data.py on dataset generated from step4. This step is used to construct vocabulary.

#### Running Instruction
1. Open Math.ipynb
2. Upload dataset and code (exclude preprocess part) into same folder
3. Run Math.ipynb
