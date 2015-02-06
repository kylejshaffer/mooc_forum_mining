'''
This is an early script used for reading the initially distributed
dataset (an Excel file), processing and keeping only the data needed
for this present analysis, and formatting as JSON.
'''

import xlrd
from xlrd import open_workbook, cellname

book = open_workbook('trips.xlsx')
sheet = book.sheet_by_index(0)

keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]
dlist = []

for row_index in range(1, sheet.nrows):
	d = {keys[col_index]: sheet.cell(row_index, col_index).value
		for col_index in range(sheet.ncols)}
	dlist.append(d)
