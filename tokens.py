# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:08:18 2019

@author: Tiago
"""

class tokens_table(object):
    def __init__(self):
        
        tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', ' ']
        
        self.table = tokens
        self.table_len = len(self.table)