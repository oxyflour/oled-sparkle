#!/usr/bin/env python3
"""Regenerate a portable HTML report without re-running optical calculations."""
import argparse
from reporting import write_html_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Existing results directory')
    parser.add_argument('--out', help='HTML destination; default: <results>/report.html')
    parser.add_argument('--recursive', action='store_true', help='Combine nested run directories')
    parser.add_argument('--title', default='OLED / AG 闪点仿真报告')
    parser.add_argument('--figure', action='append', default=[], help='Embed a comparison PNG; repeat for multiple figures')
    args = parser.parse_args()
    output = write_html_report(args.results, args.out, args.recursive, args.title, args.figure)
    print('HTML report:', output)


if __name__ == '__main__':
    main()
