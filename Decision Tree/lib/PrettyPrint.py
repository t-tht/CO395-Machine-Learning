from lib.Node import Node
import os


# Print the tree in latex and convert to pdf
def PrettyPrint(root):

    header = r"""
\documentclass{standalone}
\usepackage{forest}
\begin{document}
\begin{forest}"""

    end = r"""
\end{forest}
\end{document}
           """
    f = open("visual_tree.tex", "w+")
    f.write(header)
    #f.write(r"  \node {root}")
    root.print_tree(0, f)
    #f.write(";\n")
    f.write(end)
    f.close()

    os.system("pdflatex visual_tree.tex")
