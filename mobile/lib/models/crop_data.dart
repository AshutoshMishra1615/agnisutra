import 'package:flutter/material.dart';
import 'package:hive/hive.dart';

part 'crop_data.g.dart';

@HiveType(typeId: 0)
class CropData {
  @HiveField(0)
  final String name;

  @HiveField(1)
  final int statusColorValue; // Store color as int for Hive

  @HiveField(2)
  final double progress;

  @HiveField(3)
  final String moisture;

  @HiveField(4)
  final String temp;

  @HiveField(5)
  final String sownDate;

  @HiveField(6)
  final String lastIrrigation;

  @HiveField(7)
  final String lastPesticide;

  @HiveField(8)
  final String expectedYield;

  @HiveField(9)
  final double? latitude;

  @HiveField(10)
  final double? longitude;

  CropData({
    required this.name,
    required Color statusColor,
    required this.progress,
    required this.moisture,
    required this.temp,
    required this.sownDate,
    required this.lastIrrigation,
    required this.lastPesticide,
    required this.expectedYield,
    this.latitude,
    this.longitude,
  }) : statusColorValue = statusColor.value;

  Color get statusColor => Color(statusColorValue);
}
