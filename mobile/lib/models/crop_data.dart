import 'package:flutter/material.dart';

class CropData {
  final String name;
  final Color statusColor;
  final double progress;
  final String moisture;
  final String temp;
  final String sownDate;
  final String lastIrrigation;
  final String lastPesticide;
  final String expectedYield;

  CropData({
    required this.name,
    required this.statusColor,
    required this.progress,
    required this.moisture,
    required this.temp,
    required this.sownDate,
    required this.lastIrrigation,
    required this.lastPesticide,
    required this.expectedYield,
  });
}
