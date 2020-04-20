#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `vec2rec` package."""

from unittest import TestCase
from vec2rec.cli import main

import sys
from contextlib import contextmanager
from io import StringIO

@contextmanager
def capture_print():
    """
        A context manager to override sys.stdout with StringIO() file string comparison testing.
    :return:
    """
    _stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = _stdout


class TestTestTools(TestCase):
    """ tests tools in testtools.py """

    def test_capture_print(self):
        """ tests capture_print() """
        expected = "Name of arg is ['m', 'y', 'a', 'r', 'g', 's']\n"
        with capture_print() as std:
            main(args="myargs")
            std.seek(0)
            captured = std.readline()
            self.assertEqual(expected, captured)
