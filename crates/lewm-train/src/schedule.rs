//! Learning-rate schedules for training.

/// RFC 0005 default peak learning rate.
pub const DEFAULT_LR_PEAK: f64 = 3e-4;

/// RFC 0005 default final learning rate.
pub const DEFAULT_LR_MIN: f64 = 1e-5;

/// RFC 0005 default warmup length in optimizer steps.
pub const DEFAULT_WARMUP_STEPS: u32 = 1_000;

/// RFC 0005 default training epoch count.
pub const DEFAULT_EPOCHS: u32 = 10;

/// Learning-rate schedule defaults from RFC 0005.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScheduleConfig {
    /// Peak learning rate reached at `warmup_steps`.
    pub lr_peak: f64,

    /// Final learning rate reached at `total_steps`.
    pub lr_min: f64,

    /// Number of linear warmup optimizer steps.
    pub warmup_steps: u32,

    /// Default epoch count used by canonical training configs.
    pub epochs: u32,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ScheduleConfig {
    /// Creates a config with RFC 0005 defaults.
    pub const fn new() -> Self {
        Self {
            lr_peak: DEFAULT_LR_PEAK,
            lr_min: DEFAULT_LR_MIN,
            warmup_steps: DEFAULT_WARMUP_STEPS,
            epochs: DEFAULT_EPOCHS,
        }
    }

    /// Overrides the peak learning rate.
    pub const fn with_lr_peak(mut self, lr_peak: f64) -> Self {
        self.lr_peak = lr_peak;
        self
    }

    /// Overrides the final learning rate.
    pub const fn with_lr_min(mut self, lr_min: f64) -> Self {
        self.lr_min = lr_min;
        self
    }

    /// Overrides the warmup length.
    pub const fn with_warmup_steps(mut self, warmup_steps: u32) -> Self {
        self.warmup_steps = warmup_steps;
        self
    }

    /// Overrides the default epoch count.
    pub const fn with_epochs(mut self, epochs: u32) -> Self {
        self.epochs = epochs;
        self
    }
}

/// Linear warmup followed by cosine decay.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosineWarmup {
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u32,
    total_steps: u32,
}

impl CosineWarmup {
    /// Creates a schedule with RFC 0005 defaults for a known number of steps.
    pub const fn new(total_steps: u32) -> Self {
        Self::from_config(ScheduleConfig::new(), total_steps)
    }

    /// Creates a schedule from explicit config and total steps.
    pub const fn from_config(config: ScheduleConfig, total_steps: u32) -> Self {
        Self {
            lr_peak: config.lr_peak,
            lr_min: config.lr_min,
            warmup_steps: config.warmup_steps,
            total_steps,
        }
    }

    /// Creates a schedule from explicit scalar values.
    pub const fn from_parts(
        lr_peak: f64,
        lr_min: f64,
        warmup_steps: u32,
        total_steps: u32,
    ) -> Self {
        Self {
            lr_peak,
            lr_min,
            warmup_steps,
            total_steps,
        }
    }

    /// Returns the learning rate for an optimizer step.
    pub fn lr(&self, step: u32) -> f64 {
        if self.total_steps == 0 || step >= self.total_steps {
            return self.lr_min;
        }

        if step == self.warmup_steps {
            return self.lr_peak;
        }

        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.lr_peak * f64::from(step) / f64::from(self.warmup_steps);
        }

        let cooldown_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let cooldown_step = step.saturating_sub(self.warmup_steps);
        let progress = f64::from(cooldown_step) / f64::from(cooldown_steps);
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());

        self.lr_min + ((self.lr_peak - self.lr_min) * cosine)
    }

    /// Returns the peak learning rate.
    pub const fn lr_peak(&self) -> f64 {
        self.lr_peak
    }

    /// Returns the final learning rate.
    pub const fn lr_min(&self) -> f64 {
        self.lr_min
    }

    /// Returns the warmup length.
    pub const fn warmup_steps(&self) -> u32 {
        self.warmup_steps
    }

    /// Returns the total training step count.
    pub const fn total_steps(&self) -> u32 {
        self.total_steps
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const TOTAL_STEPS: u32 = 10_000;

    #[test]
    fn cosine_schedule_defaults_match_rfc_0005() {
        let config = ScheduleConfig::new();

        assert_eq!(config.lr_peak.to_bits(), DEFAULT_LR_PEAK.to_bits());
        assert_eq!(config.lr_min.to_bits(), DEFAULT_LR_MIN.to_bits());
        assert_eq!(config.warmup_steps, DEFAULT_WARMUP_STEPS);
        assert_eq!(config.epochs, DEFAULT_EPOCHS);
    }

    #[test]
    fn cosine_schedule_endpoints_exact() {
        let schedule = CosineWarmup::new(TOTAL_STEPS);

        assert_eq!(
            schedule.lr(DEFAULT_WARMUP_STEPS).to_bits(),
            DEFAULT_LR_PEAK.to_bits()
        );
        assert_eq!(schedule.lr(TOTAL_STEPS).to_bits(), DEFAULT_LR_MIN.to_bits());
        assert_eq!(
            schedule.lr(TOTAL_STEPS + 1).to_bits(),
            DEFAULT_LR_MIN.to_bits()
        );
    }

    #[test]
    fn cosine_schedule_linear_warmup_starts_at_zero() {
        let schedule = CosineWarmup::new(TOTAL_STEPS);

        assert_eq!(schedule.lr(0).to_bits(), 0.0_f64.to_bits());
        assert!((schedule.lr(DEFAULT_WARMUP_STEPS / 2) - (DEFAULT_LR_PEAK / 2.0)).abs() <= 1e-16);
    }

    #[test]
    fn cosine_schedule_zero_warmup_starts_at_peak() {
        let schedule = CosineWarmup::from_parts(DEFAULT_LR_PEAK, DEFAULT_LR_MIN, 0, TOTAL_STEPS);

        assert_eq!(schedule.lr(0).to_bits(), DEFAULT_LR_PEAK.to_bits());
    }

    proptest! {
        #[test]
        fn cosine_schedule_cooldown_is_monotonic(
            warmup_steps in 0_u32..1_000,
            cooldown_steps in 1_u32..5_000,
            offset in 0_u32..5_000,
        ) {
            let total_steps = warmup_steps + cooldown_steps;
            let offset = offset % cooldown_steps;
            let step = warmup_steps + offset;
            let next_step = step + 1;
            let schedule = CosineWarmup::from_parts(
                DEFAULT_LR_PEAK,
                DEFAULT_LR_MIN,
                warmup_steps,
                total_steps,
            );

            prop_assert!(schedule.lr(next_step) <= schedule.lr(step) + 1e-15);
        }
    }
}
