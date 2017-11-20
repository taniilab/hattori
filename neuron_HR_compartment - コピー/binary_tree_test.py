# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:15:06 2017

@author: Hattori
"""

recursion_cycle = 0


recursion_depth = 10

class BinaryTree:
    def __init__(self, label, left=None, right=None):
        self.label = label
        self.left = left
        self.right = right


def traverserPreorder(binarytree):
    global recursion_cycle
    recursion_cycle += 1
    if binarytree ==None:
        return;
    print(str(binarytree.label)+' passed')
    print('recursion_cycle:' + str(recursion_cycle))
    if recursion_cycle < recursion_depth:
        traverserPreorder(binarytree.left)
        traverserPreorder(binarytree.right)


def main():
    axon8 = BinaryTree(8)
    axon7 = BinaryTree(7)
    axon6 = BinaryTree(6)
    axon5 = BinaryTree(5)
    axon4 = BinaryTree(4, axon7, axon8)
    axon3 = BinaryTree(3, axon5, axon6)
    axon2 = BinaryTree(2, axon4)
    axon1 = BinaryTree(1, axon2, axon3)
    soma = BinaryTree(0, axon1)
    
    #recurrent
    axon8.left = soma
    axon7.left = soma
    axon6.left = soma
    axon5.left = soma

    traverserPreorder(soma)


if __name__ == '__main__':
    main()