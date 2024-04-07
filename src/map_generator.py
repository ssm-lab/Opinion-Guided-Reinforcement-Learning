import logging
import os
import pandas as pd
import random
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

#Constants
FILES_PATH = 'src/files'


logging.basicConfig(format='[%(levelname)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)

def index_to_cell(row, col):
    cell =  f'{get_column_letter(col+2)}{row+2}'
    logging.debug(cell)
    return cell

def sequence_to_coordinates(sequence_number, edge):
    row = sequence_number // edge
    col = sequence_number % (edge)
    logging.debug(f'({row}, {col})')
    
    return (row, col)
    
def randomize_holes(size, frozen_tiles_ratio=0.8):
    num_holes = round(size*size*(1-frozen_tiles_ratio))-2
    logging.debug(num_holes)
    random.seed(SEED) # we can use this to generate an array of reproducible maps
    hole_cells = random.sample(range(1, size*size-1), num_holes) # encoded here that (0,0) is start and (29,29) is goal
    
    return hole_cells
    
def apply_style(sheet, cell, symbol, font, color):
    sheet[cell] = symbol
    sheet[cell].font = font
    sheet[cell].fill = color
    
    thin = Side(border_style="thin", color="000000")
    sheet[cell].border = Border(top=thin, left=thin, right=thin, bottom=thin)

def create_hole(sheet, cell):
    symbol = 'H'
    font = Font(color='FFFFFF', bold=True)
    color = PatternFill(start_color='000000', end_color='000000', fill_type='solid')
    
    apply_style(sheet, cell, symbol, font, color)
    
    
def create_ice(sheet, cell):
    symbol = 'F'
    font = Font(color='000000')
    color = PatternFill(start_color='DAE9F8', end_color='DAE9F8', fill_type='solid')
    
    apply_style(sheet, cell, symbol, font, color)
    
def create_start(sheet, cell):
    symbol = 'S'
    font = Font(color='000000', bold=True)
    color = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
    
    apply_style(sheet, cell, symbol, font, color)
    
def create_goal(sheet, cell):
    symbol = 'S'
    font = Font(color='000000', bold=True)
    color = PatternFill(start_color='83E28E', end_color='83E28E', fill_type='solid')
    
    apply_style(sheet, cell, symbol, font, color)

def generate_map():
    template_file = os.path.abspath(f'{FILES_PATH}/lake-template.xlsx')
    final_file = os.path.abspath(f'{FILES_PATH}/lake.xlsx')
    workbook = load_workbook(filename=template_file)
    
    sheet = workbook.active
    holes = randomize_holes()
    for hole in holes:
        (row, col) = sequence_to_coordinates(hole)
        create_hole(sheet, index_to_cell(row, col))
    
    workbook.save(filename=final_file)

def generate_map_from_scratch(filename, size):
    file = os.path.abspath(f'{FILES_PATH}/{filename}.xlsx')
    workbook = Workbook(file)
    
    workbook.save(filename=file)
    
    workbook = load_workbook(filename=file)
    sheet = workbook.active
    
    alignment = Alignment(horizontal='center', vertical='center')
    
    for i in range(size+1):
        sheet.row_dimensions[i+1].height = 15
        sheet.column_dimensions[get_column_letter(i+1)].width = 2.75
        
    for i in range(size):
        cell = f'{get_column_letter(i+2)}1'
        sheet[cell] = i
        
    for i in range(size):
        cell = f'{get_column_letter(1)}{i+2}'
        sheet[cell] = i
        
    for i in range(size+1):
        for j in range(size+1):
            cell = f'{get_column_letter(i+1)}{j+1}'
            sheet[cell].alignment = alignment
    
    for i in range(1, size+1):
        for j in range(1, size+1):
            cell = f'{get_column_letter(i+1)}{j+1}'
            create_ice(sheet, cell)
    
    create_start(sheet, index_to_cell(0, 0))
    create_goal(sheet, index_to_cell(size-1, size-1))
    
    holes = randomize_holes(size=8)
    for hole in holes:
        (row, col) = sequence_to_coordinates(hole, 8)
        create_hole(sheet, index_to_cell(row, col))
    
    workbook.save(filename=file)
    
'''''''''''''''''''''''''''''''''''''''''''''
Main
'''''''''''''''''''''''''''''''''''''''''''''
SEED = 10
generate_map_from_scratch(f'lake-{SEED}', 8)