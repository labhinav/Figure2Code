# Figure2Code

Steps to evaluate models, assuming your generated codes are saved in a csv

Remember to set input output file and folder paths correctly for each step.

1. Run clean_code.py
2. Run filter_code.py
3. Run inference_code_to_image.py
4. Run make_images_square.py on the images generated above
5. Run extract_values.py with search_string = 'values'
6. Run extract_values.py with search_string = 'categories'
7. Run eval.py