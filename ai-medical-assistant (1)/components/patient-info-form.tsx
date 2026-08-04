'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { User } from 'lucide-react'

interface PatientInfo {
  name: string
  age: number
  gender: string
  weight: number
  height: number
  bloodGroup: string
  emergencyContact: string
}

interface PatientInfoFormProps {
  patientInfo: PatientInfo
  onPatientInfoChange: (info: PatientInfo) => void
}

export function PatientInfoForm({ patientInfo, onPatientInfoChange }: PatientInfoFormProps) {
  const updatePatientInfo = (field: keyof PatientInfo, value: string | number) => {
    onPatientInfoChange({
      ...patientInfo,
      [field]: value
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <User className="w-5 h-5 mr-2" />
          Patient Information
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label htmlFor="name">Full Name</Label>
          <Input
            id="name"
            value={patientInfo.name}
            onChange={(e) => updatePatientInfo('name', e.target.value)}
            placeholder="Enter patient's full name"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label htmlFor="age">Age</Label>
            <Input
              id="age"
              type="number"
              min="0"
              max="120"
              value={patientInfo.age}
              onChange={(e) => updatePatientInfo('age', parseInt(e.target.value) || 0)}
            />
          </div>
          <div>
            <Label htmlFor="gender">Gender</Label>
            <Select value={patientInfo.gender} onValueChange={(value) => updatePatientInfo('gender', value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Male">Male</SelectItem>
                <SelectItem value="Female">Female</SelectItem>
                <SelectItem value="Other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label htmlFor="weight">Weight (kg)</Label>
            <Input
              id="weight"
              type="number"
              min="0"
              max="300"
              value={patientInfo.weight}
              onChange={(e) => updatePatientInfo('weight', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div>
            <Label htmlFor="height">Height (cm)</Label>
            <Input
              id="height"
              type="number"
              min="0"
              max="300"
              value={patientInfo.height}
              onChange={(e) => updatePatientInfo('height', parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>

        <div>
          <Label htmlFor="bloodGroup">Blood Group</Label>
          <Select value={patientInfo.bloodGroup} onValueChange={(value) => updatePatientInfo('bloodGroup', value)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="A+">A+</SelectItem>
              <SelectItem value="A-">A-</SelectItem>
              <SelectItem value="B+">B+</SelectItem>
              <SelectItem value="B-">B-</SelectItem>
              <SelectItem value="AB+">AB+</SelectItem>
              <SelectItem value="AB-">AB-</SelectItem>
              <SelectItem value="O+">O+</SelectItem>
              <SelectItem value="O-">O-</SelectItem>
              <SelectItem value="Unknown">Unknown</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="emergencyContact">Emergency Contact</Label>
          <Input
            id="emergencyContact"
            value={patientInfo.emergencyContact}
            onChange={(e) => updatePatientInfo('emergencyContact', e.target.value)}
            placeholder="Phone number"
          />
        </div>
      </CardContent>
    </Card>
  )
}
