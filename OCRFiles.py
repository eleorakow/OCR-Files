import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from pytesseract import Output
import math

# GIMP
# Configurations for the classes
# User to edit the tesseract and poppler path according to where they saved their files
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\eleorakowsk\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\Users\eleorakowsk\Downloads\poppler-24.07.0\Library\bin"
digit_config = r"--oem 3 --psm 6 outputbase digits"
text_config = r"--oem 3 --psm 3"

import json
class ExtractTemplateCoordinates():
    def __init__(self, filepath):
        self.coords_filepath = filepath

    def read_json(self):
        with open(self.coords_filepath, 'r') as f:
            coords = json.load(f)
        df = pd.DataFrame(coords)
        df = df[['fname', 'x', 'y', 'h', 'w']]

        self.df = df
        return df
    

class FileCropper():
    def __init__(self, filepath):
        '''
        filepath takes in the user's file path for their png, jpeg, pdf, etc.
        '''
        # validate file path extension
        # include poppler path
        self.filepath = filepath
        self.page_dicts = {}
        
    def convert_to_PIL(self):
        # import pdf2image to show which package it comes from
        # explain how to install and use poppler
        # if multiple pages then user will generate a list of the different pages
        pages = convert_from_path(self.filepath, dpi=600, poppler_path=poppler_path)
        self.pages = pages
        return pages
    
    '''
    extract the information from the PIL with tesseract
    page_num is a base-1 integer
    '''
    def get_data_dict(self, page_num):
        # check if the page data dict already exists
        if page_num in self.page_dicts:
            return self.page_dicts.get(page_num)
        
        # if it dosent exist then read and add it to the dictionary
        else:
            page = self.pages[page_num - 1]
            d = pytesseract.image_to_data(page, output_type=Output.DICT)
            dict_df = pd.DataFrame(d)
            
            # remove rows that do not contain values
            dict_df = dict_df[dict_df['conf'] != -1]

            # find right and bottom
            dict_df['right'] = dict_df['left'] + dict_df['width']
            dict_df['bottom'] = dict_df['top'] + dict_df['height']

            # create a column for bounding boxes
            dict_df['bounding box'] = list(zip(dict_df['left'], dict_df['top'], dict_df['right'], dict_df['bottom']))

            # add to the dictionary of page_dicts for easy accessibility
            self.page_dicts[page_num] = dict_df
            return dict_df

    '''
    offer 4 different crop modes: 'max' - take the maximum bounds for left, top, right, bottom
                                  'anchor' - take in a compulsory anchor for the top-left and optional anchor for the bottom-right and crop
                                  'dimension' - crop based on (left, top, right, bottom) pixel bounds     **the only crop method that does not involve running through Tesseract
                                  'no' - don't crop

    example of crop_input: {1: ('no'), 3: ('anchor', ['anchor1', anchor2']), 7: ('dimension', (0, 1, 2, 3))}
    crop_input allows user to specify only the pages they would like to crop and allows for different crop modes for different pages                              
    '''

    def crop_pages(self, crop_input, specification):
        cropped_pages = []
        for page_num in crop_input:
            cropped_page = self.crop_page(page_num, crop_input.get(page_num))
            cropped_pages.append(cropped_page)

        self.cropped_pages = cropped_pages
        return cropped_pages

    def crop_page(self, page_num, crop_input):
        cur_page = self.pages[page_num - 1]
        crop_mode = crop_input[0]

        # crop_input example: {1: ('no')}
        if crop_mode == 'no':
            return cur_page
        
        # crop_input example: {2: ('max')}
        elif crop_mode == 'max':
            # get the maximum bounds by reading through tesseract and extracting the left and top most bounds
            dict_df = self.get_data_dict(page_num)
            left_bound = dict_df['left'].min()
            top_bound = dict_df['top'].min()
            right_bound = dict_df['right'].max()
            bottom_bound = dict_df['bottom'].max()

            # crop the PIL Image
            cur_page = cur_page.crop((left_bound, top_bound, right_bound, bottom_bound))
            return cur_page
        
        # crop_input example: {3: ('anchor', ['topleft anchor', 'bottomright anchor'])}
        # user can choose to omit the bottom right anchor if they only want to crop the top and left margins
        elif crop_mode == 'anchor':
            dict_df = self.get_data_dict(page_num)
            anchors = crop_input[1]

            if len(anchors) == 1:
                topleft_anchor = anchors[0]
                topleft_df = dict_df[dict_df['text'] == topleft_anchor]

                left_bound = topleft_df['left'].min()
                top_bound = topleft_df['top'].min()

                width, height = cur_page.size
                cur_page = cur_page.crop((left_bound, top_bound, width, height))
                return cur_page
            
            else:
                topleft_anchor = anchors[0]
                bottomright_anchor = anchors[1]
                topleft_df = dict_df[dict_df['text'] == topleft_anchor]
                bottomright_df = dict_df[dict_df['text'] == bottomright_anchor]

                left_bound = topleft_df['left'].min()
                top_bound = topleft_df['top'].min()
                right_bound = bottomright_df['right'].max()
                bottom_bound = bottomright_df['bottom'].max()

                cur_page = cur_page.crop((left_bound, top_bound, right_bound, bottom_bound))
                return cur_page
        
        # crop_input example: {4: ('dimension', (0, 100, 4555, 7888))}
        elif crop_mode == 'dimension':
            crop_dimensions = crop_input[1]
            
            # crop the PIL Image
            cur_page = cur_page.crop((crop_dimensions))
            return cur_page
        
        # raise an exception if the user does not enter or enters an invalid crop mode
        else:
            raise Exception("Please enter one of the following modes: 'max', 'anchor', 'dimensions', 'no'. ")
   
'''
pages are from pdf2image.convert_from_path / run FileCropper 
explain the methodology 
'''

from scipy.spatial.distance import euclidean

class OCRCoords():
    # Take in a df which must have columns var, page_num, coords = (left, top, right, bottom), config = text / digit / base
    def __init__(self, pages, coord_df):
        self.coord_df = coord_df
        self.pages = pages
        # remind gs to remind sj to use confidence for conditional formatting when output to excel
        self.results = pd.DataFrame(columns=['output', 'conf'])
        self.digit_config = r"--oem 3 --psm 6 outputbase digits"
        self.text_config = r"--oem 3 --psm 3"

    def find_midpoint(self, df):
        df['midpoint'] = list(zip((df['left'] + df['right']) / 2, (df['top'] + df['bottom']) / 2))
        return df
    
    def find_midpoint(self, coords):
        left, top, right, bottom = coords
        midpoint_x = (left + right) / 2
        midpoint_y = (top + bottom) / 2
        return (midpoint_x, midpoint_y)
    
    def calculate_overlap(self, box1, box2):
        left1, top1, right1, bottom1 = box1
        left2, top2, right2, bottom2 = box2

        # calculate the overlap coordinates
        overlap_left = max(left1, left2)
        overlap_top = max(top1, top2)
        overlap_right = min(right1, right2)
        overlap_bottom = min(bottom1, bottom2)

        # calculate the overlapping region's width and height
        overlap_width = max(0, overlap_right - overlap_left)
        overlap_height = max(0, overlap_bottom - overlap_top)

        overlap_area = overlap_width * overlap_height
        return overlap_area

# change scipy.spatial.distance.euclidean
    def calculate_euclideanDist(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
    
    # change to vectorised calculations
    def increase_bounds(self, df, buffer=0.2):
        # calculate height and width based on the coordinates given
        df['width'] = df['right'] - df['left']
        df['height'] = df['bottom'] - df['top']

        # find the change in width and height
        df['width'] = (buffer * df['width']) / 2
        df['height'] = (buffer * df['height']) / 2

        # add a column for the increased bounds
        df['increased bounds'] = list(zip(df['left'] - df['width'], df['top'] - df['height'], df['right'] + df['width'], df['bottom'] + df['height']))
        return df
    
    def readOCR(self):
        # increase bounds and find midpoint for each variable in the coord_df
        coord_df = self.coord_df
        coord_df = self.increase_bounds(coord_df)
        coord_df = self.find_midpoint(coord_df)
        results = pd.DataFrame(columns=['output', 'conf'])

        for input_index, input_row in coord_df.iterrows():
            page = input_row['page_num']
            original_boundingbox = input_row['bounding box']
            increased_bounds = input_row['increaesd bounds']
            config = input_row['config']
            cur_image = self.pages[page - 1]

            # crop according to the increased bounds
            cur_image = cur_image.crop(increased_bounds)

            # read and extract data with pytesseract 
            d = pytesseract.image_to_data(cur_image, output_type=Output.DICT, config=config)
            d_df = pd.DataFrame(d)
            d_df = d_df[d_df['conf'] != -1]

            # calculate the midpoint for each piece of data 
            d_df = self.find_midpoint(d_df)
            
            d_df['euclidean dist'] = self.calculate_euclideanDist(d_df['midpoint'], input_row['midpoint'])
            closest_df = d_df.sort_values(by='euclidean dist', ascending=True).head(3)

            # cannot take the closest as it might not have the greatest overlap with the original coordinates specified
            output_text = ""
            conf = 0
            max_overlap = 0
            for potential_index, potential_row in closest_df.iterrows():
                cur_overlap = self.calculate_overlap(potential_row['bounding box'], original_boundingbox)
                if cur_overlap > max_overlap:
                    max_overlap = cur_overlap
                    conf = potential_row['conf']
                    output_text = potential_row['text']

            results.loc[len(results)] = [output_text, conf]

        self.results = results
        output_to_user = pd.concat([self.coord_df, results], ignore_index=True)
        return output_to_user