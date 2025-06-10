Available preprocessing functions:
1. document_type
2. document_type_cleaner

Enter the number of the function you want to run (or 'q' to quit): 1

Running document_type...
Enter the path to your SQLite database: database.db
Number of records with null Document Type: 0

Document Types and their counts:
Article: 154071
Conference paper: 34840
Review: 18980
Book chapter: 3894
Editorial: 3136
Conference review: 2618
Note: 2396
Letter: 2273
Short survey: 820
Erratum: 567
Retracted: 301
Book: 289
Data paper: 104
Article in press: 5
Report: 1
: 1

Enter the number of the function you want to run (or 'q' to quit): 2

Running document_type_cleaner...
Enter the path to your SQLite database: database.db
Remaining Document Types and their counts:
Article: 154071

Enter the number of the function you want to run (or 'q' to quit): q

Process finished with exit code 0


"C:\Program Files\Python312\python.exe" "C:/Users/V/Documents/Reza Projects/cancer-ai-trend-py/main.py" 
Available functions:
Preprocessing:
P1. document_type
P2. document_type_cleaner
P3. title_abstract_cleaner

Analysis:
A1. chech_pkl_file

Enter the number of the function you want to run (prefix P for preprocessing, A for analysis, or 'q' to quit): A1

Running chech_pkl_file from analysis...
Enter the name of the PKL file (including .pkl extension): processed_data.pkl

Loaded Data Type: <class 'pandas.core.frame.DataFrame'>
Detected Pandas DataFrame, showing first 5 rows:
                                               Title  ...                             Cleaned_Index Keywords
0  Optical devices used for image analysis of pig...  ...  biophysic human image processing computer assi...
1  Estrogen receptor α target genes: The role of ...  ...  estrogen receptor alpha article breast cancer ...
2  Needle and seed segmentation in intra-operativ...  ...  algorithms animal artificial intelligence brac...
3  Heavy incense burning in temples promotes expo...  ...  air pollutants air pollution indoor carcinogen...
4  Premeiotic germ cell defect in seminiferous tu...  ...  animal apoptosis cell aging cell cycle cell cy...

[5 rows x 8 columns]

DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 154071 entries, 0 to 154070
Data columns (total 8 columns):
 #   Column                   Non-Null Count   Dtype 
---  ------                   --------------   ----- 
 0   Title                    154071 non-null  object
 1   Abstract                 154071 non-null  object
 2   Author Keywords          154071 non-null  object
 3   Index Keywords           154071 non-null  object
 4   Cleaned_Title            154071 non-null  object
 5   Cleaned_Abstract         154071 non-null  object
 6   Cleaned_Author Keywords  154071 non-null  object
 7   Cleaned_Index Keywords   154071 non-null  object
dtypes: object(8)
memory usage: 9.4+ MB
None

Column Names: ['Title', 'Abstract', 'Author Keywords', 'Index Keywords', 'Cleaned_Title', 'Cleaned_Abstract', 'Cleaned_Author Keywords', 'Cleaned_Index Keywords']
Total Records: 154071

Enter the number of the function you want to run (prefix P for preprocessing, A for analysis, or 'q' to quit): q

Process finished with exit code 0